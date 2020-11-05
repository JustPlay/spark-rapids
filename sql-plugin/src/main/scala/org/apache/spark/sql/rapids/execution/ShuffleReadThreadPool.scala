/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids

import java.util.{Collections, Locale}
import java.util.concurrent._
import java.util.concurrent.atomic._

import scala.annotation.tailrec
import scala.collection.JavaConverters._
import scala.collection.immutable.HashSet
import scala.collection.mutable.{ArrayBuffer, LinkedHashMap, Queue}
import scala.math.max

import com.google.common.util.concurrent.ThreadFactoryBuilder

import org.apache.spark.TaskContext
import org.apache.spark.SparkEnv
import org.apache.spark.internal.Logging
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}

// Singleton threadpool that is used across all the tasks in this executor.
// Please note that the TaskContext is not set in these threads and should not be used.
object ShuffleFetchThreadPool extends Logging {
  private var threadPool: Option[ThreadPoolExecutor] = None

  private def initialize(maxThreads: Int, keepAliveSeconds: Long = 60): ThreadPoolExecutor = synchronized {
    if (!threadPool.isDefined) {
      logInfo(s"Initialize background shuffle fetcher(s), maxThreds=$maxThreads, keepAliveSeconds=$keepAliveSeconds")
      val threadFactory = new ThreadFactoryBuilder()
          .setNameFormat("rapids shuffle background fetcher-%d")
          .setDaemon(true)
          .build()

      threadPool = Some(new ThreadPoolExecutor(
          maxThreads, // corePoolSize: max number of threads to create before queuing the tasks
          maxThreads, // maximumPoolSize: because we use LinkedBlockingDeque, this is not used
          keepAliveSeconds,
          TimeUnit.SECONDS,
          new LinkedBlockingQueue[Runnable],
          threadFactory))

      threadPool.get.allowCoreThreadTimeOut(true)
    }

    threadPool.get
  }

  def submit[T](task: Callable[T], numThreads: Int = 8): Future[T] = {
    threadPool.getOrElse(initialize(numThreads)).submit(task)
  }
}

object ShuffleBackgroundFetcher {
  var rapidsConf: Option[RapidsConf] = synchronized {
    Some(new RapidsConf(SparkEnv.get.conf))
  }

  val gpuTargetBatchSizeBytes = {
    val conf = rapidsConf.getOrElse(new RapidsConf(SparkEnv.get.conf))
    conf.gpuTargetBatchSizeBytes
  }

  val pinnedPoolSize = {
    val conf = rapidsConf.getOrElse(new RapidsConf(SparkEnv.get.conf))
    if (conf.pinnedPoolSize > 0) {
      conf.pinnedPoolSize
    } else {
      Long.MaxValue
    }
  }

  // soft limit on the maximum size in byte of the data in bufferPool
  val maxSizeOfBufferPool: Long = Math.min(gpuTargetBatchSizeBytes * 2, pinnedPoolSize)
}

/**
 * The shuffle background fetch for a spark-task
 *
 * @param iter the upstream iterator
 * @param maxThreads the size of the threadpool
 */
 class ShuffleBackgroundFetcher(iter: Iterator[ColumnarBatch], maxThreads: Int) extends Iterator[ColumnarBatch] with Logging {
  // whether the fetch has finished
  private var fetchDone = new AtomicBoolean(false)
  def isDone: Boolean = fetchDone.get()
  def setDone(): Unit = fetchDone.set(true)

  // whether there exists a background fetcher for this spark task
  private var hasBackgroundFetcher = new AtomicBoolean(false)
  def startFetcher(): Unit = hasBackgroundFetcher.set(true)
  def stopFetcher(): Unit = hasBackgroundFetcher.set(false)
  def hasFetcher: Boolean = hasBackgroundFetcher.get()
  
  // the buffer between background fetcher and the iterator
  private val bufferPool = new ConcurrentLinkedQueue[ColumnarBatch]
  
  // the current size of data in the bufferPool
  private val currentSizeOfBufferPool = new AtomicLong(0L)

  def incCurrentSize(size: Long) = currentSizeOfBufferPool.addAndGet(size)
  def decCurrentSize(size: Long) = currentSizeOfBufferPool.addAndGet(-size)
  def mayFetchContinue: Boolean = currentSizeOfBufferPool.get() < ShuffleBackgroundFetcher.maxSizeOfBufferPool

  val lock = new Object()

  override def hasNext: Boolean = {
    if (bufferPool.isEmpty() && !isDone) {
      if (!hasFetcher) {
        ShuffleFetchThreadPool.submit(new Fetcher(), maxThreads)
      }

      // https://stackoverflow.com/questions/5999100/is-there-a-block-until-condition-becomes-true-function-in-java
      // wait the background fetcher
      try {
        lock.synchronized {
          // the background fetcher may notify before we are going into wait, so check again
          if (bufferPool.isEmpty() && !isDone) {
            wait();
          }
        }
      } catch {
        case e: InterruptedException => logInfo("ShuffleBackgroundFetcher.iterator().hasNext interrupted!")
      }
    }

    if (!bufferPool.isEmpty()) {
      true
    } else {
      false
    }
  }

  override def next(): ColumnarBatch = {
    val cb = bufferPool.poll()
    if (cb == null) {
      throw new IllegalStateException("Nothing to fetch from bufferPool!")
    }
    
    val size = RapidsHostColumnVector.extractBases(cb).map(_.getHostMemorySize).sum
    decCurrentSize(size)

    // add a background fetcher if necessary
    if (!hasFetcher) {
      ShuffleFetchThreadPool.submit(new Fetcher(), maxThreads)
    }

    // convert the host-size ColumnarBatch to device-side ColumnarBatch
    GpuSemaphore.acquireIfNecessary(TaskContext.get())
        
    // TODO(2020-11-03): we should to do it in a more fragment-friendly way，the current code will cause GPU-memory fragmentation
    val numColumns = cb.numCols()
    val numRows = cb.numRows()
    try {
      if (numColumns > 0) {
        val gpuColumns = RapidsHostColumnVector.extractBases(cb).map(v => GpuColumnVector.from(v.copyToDevice().incRefCount()))
        val columns: Array[ColumnVector] = new Array(numColumns)
        for (i <- 0 until numColumns) {
          columns(i) = gpuColumns(i)
        }
        new ColumnarBatch(columns, numRows)
      } else {
        new ColumnarBatch(Array.empty, numRows)
      } 
    } finally {
      cb.close()
    }
  }

  class Fetcher() extends Callable[Unit] {
    override def call(): Unit = {
      startFetcher()
      
      try {
        while (mayFetchContinue && iter.hasNext) {
          val cb = iter.next()
          val size = RapidsHostColumnVector.extractBases(cb).map(_.getHostMemorySize).sum
          incCurrentSize(size)
          bufferPool.add(cb)
        
          // NOTE: we do not issue a notifyAll() until we have enough data in the bufferPool, so no notifyAll() here
        }

        if (!iter.hasNext) {
          setDone()
        }

        // signal when we finish fetch
        lock.synchronized {
          notifyAll()
        }
      } finally {
        stopFetcher()
      }
    }
  }
}
