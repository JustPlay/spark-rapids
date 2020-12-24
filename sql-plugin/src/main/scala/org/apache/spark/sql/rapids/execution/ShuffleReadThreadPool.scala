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

import com.nvidia.spark.rapids.{RapidsHostColumnVector, RapidsHostColumnVectorFromBuffer, GpuColumnVector}
import com.nvidia.spark.rapids.{RapidsConf, GpuSemaphore}

import ai.rapids.cudf.{HostTable, Table}

import org.apache.spark.TaskContext
import org.apache.spark.SparkEnv
import org.apache.spark.SparkConf
import org.apache.spark.internal.Logging
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}

// Singleton threadpool that is used across all the tasks in this executor.
object ShuffleFetchThreadPool extends Logging {
  private var threadPool: Option[ThreadPoolExecutor] = None

  private def initialize(maxThreads: Int, keepAliveSeconds: Long = 60): ThreadPoolExecutor = synchronized {
    if (!threadPool.isDefined) {
      logInfo(s"initialize shuffle fetch thread pool, maxThreads=$maxThreads, keepAliveSeconds=$keepAliveSeconds")
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
  val rc: Option[RapidsConf] = synchronized {
    Some(new RapidsConf(SparkEnv.get.conf))
  }

  val gpuTargetBatchSizeBytes: Long = rc.get.gpuTargetBatchSizeBytes

  val pinnedPoolSize: Long = {
    if (rc.get.pinnedPoolSize > 0) {
      rc.get.pinnedPoolSize
    } else {
      16 * 1024 * 1024 * 1024 // 16G
    }
  }

  val executorCores: Int = 8 // TODO(2020-12-20), we need a workable method to obtain `spark.executor.cores`

  // soft limit on the maximum size in byte of the data in bufferPool
  val maxSizeOfBufferPool: Long = Math.ceil(0.5 * pinnedPoolSize / executorCores).toLong

  val minSizeToDeliverToDownstream: Long = gpuTargetBatchSizeBytes
}

/**
 * The shuffle background fetcher for a spark task
 *
 * @param upstreamIter the upstream iterator
 * @param maxThreads the size of the threadpool
 */
 class ShuffleBackgroundFetcher(upstreamIter: Iterator[ColumnarBatch]) extends Iterator[ColumnarBatch] with Logging {
  val maxThreads: Int = ShuffleBackgroundFetcher.executorCores

  val taskContext = TaskContext.get()
  val taskAttemptId = taskContext.taskAttemptId()
  val stageId = taskContext.stageId()
  logInfo(s"shuffle fetcher for stage=$stageId, task=$taskAttemptId, poolsize=${ShuffleBackgroundFetcher.maxSizeOfBufferPool} bytes")

  private val _done = new AtomicBoolean(false)
  def isDone: Boolean = _done.get()
  def done(): Unit = _done.set(true)

  private val _fetcher = new AtomicBoolean(false)
  def start(): Unit = _fetcher.set(true)
  def stop(): Unit = _fetcher.set(false)
  def isFetcherActive: Boolean = _fetcher.get()
  
  private val bufferPool = new ConcurrentLinkedQueue[ColumnarBatch]
  
  private val _poolSize = new AtomicLong(0L)
  def incSize(size: Long) = _poolSize.addAndGet(size)
  def decSize(size: Long) = _poolSize.addAndGet(-size)
  def poolSize() = _poolSize.get()
  def mayFetch: Boolean = _poolSize.get() < ShuffleBackgroundFetcher.maxSizeOfBufferPool
  def maySignal: Boolean = _poolSize.get() >= ShuffleBackgroundFetcher.minSizeToDeliverToDownstream

  var numBatchsDeliveredToGpu: Int = 0
  var numBatchsFromUpstream: Int = 0
  var totalSize: Long = 0

  val lock = new Object()

  override def hasNext: Boolean = {
    if (bufferPool.isEmpty() && !isDone) {
      if (!isFetcherActive) {
        ShuffleFetchThreadPool.submit(new Fetcher(), maxThreads)
      }

      try {
        // the background fetcher may notify before we are going into wait, so check again in the lock guard
        lock.synchronized {
          if (bufferPool.isEmpty() && !isDone) {
            lock.wait();
          }
        }
      } catch {
        case e: InterruptedException => logInfo("ShuffleBackgroundFetcher.iterator().hasNext() interrupted!")
      }
    }
  
    bufferPool.isEmpty() == false
  }

  override def next(): ColumnarBatch = {
    val columnarBatchOnHost = bufferPool.poll()
    
    val numCols = columnarBatchOnHost.numCols()
    val numRows = columnarBatchOnHost.numRows()
    
    val size = RapidsHostColumnVector.extractBases(columnarBatchOnHost).map(_.getHostMemorySize).sum
    decSize(size)

    if (!isFetcherActive) {
      ShuffleFetchThreadPool.submit(new Fetcher(), maxThreads)
    }
  
    GpuSemaphore.acquireIfNecessary(TaskContext.get)
    
    var columnarBatchOnGpu = new ColumnarBatch(Array.empty, numRows)
     
    try {
      if (numCols > 0) {
        columnarBatchOnGpu = transformHostColumnarBatchToGpu(columnarBatchOnHost)
        numBatchsDeliveredToGpu = numBatchsDeliveredToGpu + 1
        totalSize = totalSize + size
      }
    } finally {
      columnarBatchOnHost.close()
    }
    
    columnarBatchOnGpu
  }

  private def transformHostColumnarBatchToGpu(columnarBatchOnHost: ColumnarBatch): ColumnarBatch = {
    val buffer = columnarBatchOnHost.column(0).asInstanceOf[RapidsHostColumnVectorFromBuffer].getBuffer()
    val columns = RapidsHostColumnVector.extractBases(columnarBatchOnHost)
    var table: Table = null
    try {
      table = HostTable.copyToDevice(buffer, columns)
      GpuColumnVector.from(table)
    } finally {
      if (table != null) {
        table.close()
      }
    }
  }

  private class Fetcher() extends Callable[Unit] with Logging {
    override def call(): Unit = {
      TaskContext.setTaskContext(taskContext)
      val id = TaskContext.get().taskAttemptId()
     
      start()
    
      try {
        while (mayFetch && upstreamIter.hasNext) {
          val cb = upstreamIter.next()
          val size = RapidsHostColumnVector.extractBases(cb).map(_.getHostMemorySize).sum
          incSize(size)
          bufferPool.add(cb)
          numBatchsFromUpstream = numBatchsFromUpstream + 1
          
          if (maySignal) {
            lock.synchronized {
              lock.notifyAll()
            }
          }
        }
      
        if (!upstreamIter.hasNext) {
          done()
        }
      } catch {
        case e: Throwable => logInfo(s"shuffle fetcher for task-$id had exception raised (" + e + ")")
      } finally {
        lock.synchronized {
          lock.notifyAll()
        }

        stop()
      }
    }
  }
}

/**
 * A Iterator wrapper which transform host-side ColumnarBatch to gpu-side ColumnarBatch
 *
 * @param upstreamIter a Iterator[ColumnarBatch] from which we can get host-side ColumnarBatch
 */
class HostToDeviceIteratorWrapper(upstreamIter: Iterator[ColumnarBatch]) extends Iterator[ColumnarBatch] with Logging {
  val context = TaskContext.get()
  val taskAttemptId = context.taskAttemptId()
  val stageId = context.stageId()

  // use `numBatch` to count how many ColumnarBatch(s) had been send to GPU by this Iterator
  var numBatch: Int = 0
  // use `totalSize' to count the data size this Iterator had send to GPU
  var totalSize: Long = 0

  override def hasNext: Boolean = {
    val status = upstreamIter.hasNext   // this will call `GpuColumnarBatchSerializer.scala:tryReadNext()` finally, so this time include the `deser` time
    status
  }   

  override def next(): ColumnarBatch = {
    val columnarBatchOnHost = upstreamIter.next()

    val numCols = columnarBatchOnHost.numCols()
    val numRows = columnarBatchOnHost.numRows()
        
    // acquire the GpuSemaphore before send data to GPU
    GpuSemaphore.acquireIfNecessary(context)

    var columnarBatchOnGpu = new ColumnarBatch(Array.empty, numRows)
     
    try {
      if (numCols > 0) {
        columnarBatchOnGpu = transformHostColumnarBatchToGpu(columnarBatchOnHost)
      } 
    } finally {
      columnarBatchOnHost.close()
    }
  
    columnarBatchOnGpu
  }

  private def transformHostColumnarBatchToGpu(columnarBatchOnHost: ColumnarBatch): ColumnarBatch = {
    val buffer = columnarBatchOnHost.column(0).asInstanceOf[RapidsHostColumnVectorFromBuffer].getBuffer()
    val columns = RapidsHostColumnVector.extractBases(columnarBatchOnHost)
    var table: Table = null
    try {
      table = HostTable.copyToDevice(buffer, columns)
      GpuColumnVector.from(table)
    } finally {
      if (table != null) {
        table.close()
      }
    }
  }
}

class TimeCountingIteratorWrapper(upstreamIter: Iterator[ColumnarBatch]) extends Iterator[ColumnarBatch] with Logging {
  val context = TaskContext.get()
  val taskAttemptId = context.taskAttemptId()
  val stageId = context.stageId()

  override def hasNext: Boolean = {
    val status = upstreamIter.hasNext
    status
  }   

  override def next(): ColumnarBatch = {
    val columnarBatchOnGpu = upstreamIter.next()
    columnarBatchOnGpu
  }
}
