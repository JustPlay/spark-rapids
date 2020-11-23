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

// In order to access protected[spark] function(s) in TaskContext.scala we need to be in a spark package
package org.apache.spark.sql.rapids.execution

import java.util.{Collections, Locale}
import java.util.concurrent._
import java.util.concurrent.atomic._

import scala.annotation.tailrec
import scala.collection.JavaConverters._
import scala.collection.immutable.HashSet
import scala.collection.mutable.{ArrayBuffer, LinkedHashMap, Queue}
import scala.math.max

import com.google.common.util.concurrent.ThreadFactoryBuilder

import com.nvidia.spark.rapids.{RapidsConf, RapidsHostColumnVector, GpuColumnVector, GpuSemaphore}

import org.apache.spark.TaskContext
import org.apache.spark.SparkEnv
import org.apache.spark.internal.Logging
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}

// Singleton threadpool that is used across all the tasks in this executor.
object ShuffleFetchThreadPool extends Logging {
  private var threadPool: Option[ThreadPoolExecutor] = None

  private def initialize(maxThreads: Int, keepAliveSeconds: Long = 60): ThreadPoolExecutor = synchronized {
    if (!threadPool.isDefined) {
      logInfo(s"initialize background shuffle fetcher(s), maxThreads=$maxThreads, keepAliveSeconds=$keepAliveSeconds")
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
  val rapidsConf: Option[RapidsConf] = synchronized {
    Some(new RapidsConf(SparkEnv.get.conf))
  }

  val gpuTargetBatchSizeBytes: Long = {
    val conf = rapidsConf.getOrElse(new RapidsConf(SparkEnv.get.conf))
    conf.gpuTargetBatchSizeBytes
  }

  val pinnedPoolSize: Long = {
    val conf = rapidsConf.getOrElse(new RapidsConf(SparkEnv.get.conf))
    if (conf.pinnedPoolSize > 0) {
      conf.pinnedPoolSize
    } else {
      Long.MaxValue
    }
  }

  // soft limit on the maximum size in byte of the data in bufferPool
  val maxSizeOfBufferPool: Long = Math.min(gpuTargetBatchSizeBytes * 4, pinnedPoolSize)
}

/**
 * The shuffle background fetcher(s) for a spark task
 *
 * @param upstreamIter the upstream iterator
 * @param maxThreads the size of the threadpool
 */
 class ShuffleBackgroundFetcher(upstreamIter: Iterator[ColumnarBatch], maxThreads: Int = 16) extends Iterator[ColumnarBatch] with Logging {
  assert(upstreamIter != null)
  
  val taskContext = TaskContext.get()
  val taskAttemptId = taskContext.taskAttemptId()
  logInfo(s"initialize background shuffle fetcher for task-$taskAttemptId, poolsize=${ShuffleBackgroundFetcher.maxSizeOfBufferPool} bytes")

  // whether the fetch has finished
  private var fetchDone = new AtomicBoolean(false)
  def isDone: Boolean = fetchDone.get()
  def setDone(): Unit = fetchDone.set(true)

  // whether there exists a background fetcher for this spark task
  private var hasBackgroundFetcher = new AtomicBoolean(false)
  def startFetcher(): Unit = hasBackgroundFetcher.set(true)
  def stopFetcher(): Unit = hasBackgroundFetcher.set(false)
  def hasFetcher: Boolean = hasBackgroundFetcher.get()
  
  // a buffer used to store ColumnarBatch(s) fetched by the background fetcher
  // the next() called by spark task will take ColumnarBatch from this buffer
  private val bufferPool = new ConcurrentLinkedQueue[ColumnarBatch]
  
  // the current size of data in the bufferPool
  private var currentSizeOfBufferPool = new AtomicLong(0L)
  def incCurrentSize(size: Long) = currentSizeOfBufferPool.addAndGet(size)
  def decCurrentSize(size: Long) = currentSizeOfBufferPool.addAndGet(-size)
  def currentPoolSize() = currentSizeOfBufferPool.get()
  def mayFetchContinue: Boolean = currentSizeOfBufferPool.get() < ShuffleBackgroundFetcher.maxSizeOfBufferPool

  var numBatchsPutOntoGpu: Int = 0
  var numBatchsFromUpstream: Int = 0

  val lock = new Object()

  override def hasNext: Boolean = {
    if (bufferPool.isEmpty() && !isDone) {
      if (!hasFetcher) {
        logInfo(s"task-$taskAttemptId will add a background fetcher (@bufferPool.isEmpty() but @isDone==false)")
        ShuffleFetchThreadPool.submit(new Fetcher(), maxThreads)
      }

      // https://stackoverflow.com/questions/5999100/is-there-a-block-until-condition-becomes-true-function-in-java
      // wait the background fetcher
      try {
        // the background fetcher may notify before we are going into wait, so check again in the lock guard
        lock.synchronized {
          if (bufferPool.isEmpty() && !isDone) {
            logInfo(s"task-$taskAttemptId need wait the background fetcher until @bufferPool.isEmpty()==false or @isDone==true")
            lock.wait();
            logInfo(s"task-$taskAttemptId has finished waiting")
          }
        }
      } catch {
        case e: InterruptedException => logInfo("ShuffleBackgroundFetcher.iterator().hasNext() interrupted!")
      }
    }

    bufferPool.isEmpty() == false
  }

  override def next(): ColumnarBatch = {
    val hostColumnarBatch = bufferPool.poll()
    if (hostColumnarBatch == null) {
      throw new IllegalStateException(s"Nothing to fetch from @bufferPool, task-$taskAttemptId")
    }
    
    val size = RapidsHostColumnVector.extractBases(hostColumnarBatch).map(_.getHostMemorySize).sum
    decCurrentSize(size)

    // add a background fetcher if necessary
    if (!hasFetcher) {
      logInfo(s"task-$taskAttemptId will add a background fetcher since the @bufferPool has free space for new batch)")
      ShuffleFetchThreadPool.submit(new Fetcher(), maxThreads)
    }
  
    GpuSemaphore.acquireIfNecessary(TaskContext.get)
    
    // TODO(2020-11-03): we should to do it in a more fragment-friendly way，the current code will cause GPU-memory fragmentation
    val numColumns = hostColumnarBatch.numCols()
    val numRows = hostColumnarBatch.numRows()
    try {
      var gpuColumnarBatch = new ColumnarBatch(Array.empty, numRows)

      if (numColumns > 0) {
        val rapidsColumns = RapidsHostColumnVector.extractBases(hostColumnarBatch).map(cv => GpuColumnVector.from(cv.copyToDevice()))
        val sparkColumns: Array[ColumnVector] = new Array(numColumns)
        for (i <- 0 until numColumns) {
          sparkColumns(i) = rapidsColumns(i)
        }
        gpuColumnarBatch = new ColumnarBatch(sparkColumns, numRows)
      }

      numBatchsPutOntoGpu = numBatchsPutOntoGpu + 1
      val size = GpuColumnVector.extractBases(gpuColumnarBatch).map(_.getDeviceMemorySize).sum
      val taskAttemptId = TaskContext.get().taskAttemptId()
      logInfo(s"task-$taskAttemptId has successfully put batch=$numBatchsPutOntoGpu onto device, size=$size bytes")

      gpuColumnarBatch
    } finally {
      hostColumnarBatch.close()
    }
  }

  private class Fetcher() extends Callable[Unit] with Logging {
    override def call(): Unit = {
      // we need pass the task's context into the background fetcher thread
      TaskContext.setTaskContext(taskContext)
      val id = TaskContext.get().taskAttemptId()
     
      startFetcher()
      logInfo(s"shuffle fetcher for task-$id started")
    
      try {
        while (mayFetchContinue && upstreamIter.hasNext) {
          val cb = upstreamIter.next()
          val size = RapidsHostColumnVector.extractBases(cb).map(_.getHostMemorySize).sum
          incCurrentSize(size)
          bufferPool.add(cb)
          numBatchsFromUpstream = numBatchsFromUpstream + 1
          // NOTE: we do not issue a notifyAll() until we have enough data in the bufferPool, so no notifyAll() here
        }
        val poolSize = currentPoolSize()
        logInfo(s"shuffle fetcher for task-$id, total $numBatchsFromUpstream batches fetched, currently total $poolSize bytes in @bufferPool")

        if (!upstreamIter.hasNext) {
          setDone()
          logInfo(s"fetch finished for spark task-$taskAttemptId, total $numBatchsFromUpstream batches fetched")
        }
      } catch {
        case e: Throwable => logInfo(s"shuffle fetcher for task-$id had exception raised (" + e + ")")
      } finally {
        // signal when we finish fetch successfully or on error
        lock.synchronized {
          logInfo(s"signal to spark task-$taskAttemptId")
          lock.notifyAll()
        }

        stopFetcher()
        logInfo(s"shuffle fetcher for task-$id ended")
      }
    }
  }
}

class HostToDeviceIteratorWrapper(upstreamIter: Iterator[ColumnarBatch]) extends Iterator[ColumnarBatch] with Logging {
  var batchId: Int = 0

  override def hasNext: Boolean = upstreamIter.hasNext
   
  override def next(): ColumnarBatch = {
    val hostColumnarBatch = upstreamIter.next()
    batchId = batchId + 1
    
    GpuSemaphore.acquireIfNecessary(TaskContext.get)
    
    // TODO(2020-11-03): we should to do it in a more fragment-friendly way，the current code will cause GPU-memory fragmentation
    val numColumns = hostColumnarBatch.numCols()
    val numRows = hostColumnarBatch.numRows()
    try {
      var gpuColumnarBatch = new ColumnarBatch(Array.empty, numRows)

      if (numColumns > 0) {
        val rapidsColumns = RapidsHostColumnVector.extractBases(hostColumnarBatch).map(cv => GpuColumnVector.from(cv.copyToDevice()))
        val sparkColumns: Array[ColumnVector] = new Array(numColumns)
        for (i <- 0 until numColumns) {
          sparkColumns(i) = rapidsColumns(i)
        }
        gpuColumnarBatch = new ColumnarBatch(sparkColumns, numRows)
      }

      val size = GpuColumnVector.extractBases(gpuColumnarBatch).map(_.getDeviceMemorySize).sum
      val taskAttemptId = TaskContext.get().taskAttemptId()
      logInfo(s"task-$taskAttemptId has successfully put batch=$batchId onto device, size=$size bytes")

      gpuColumnarBatch
    } finally {
      hostColumnarBatch.close()
    }
  }
}
