/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
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

package com.nvidia.spark.rapids;

import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.ContiguousHostTable;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.HostTable;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.vectorized.ColumnarBatch;

/** The host version of GpuColumnVectorFromBuffer */
public final class RapidsHostColumnVectorFromBuffer extends RapidsHostColumnVector {
  private final HostMemoryBuffer buffer;

  public static ColumnarBatch from(ContiguousHostTable contigTable) {
    HostMemoryBuffer buffer = contigTable.getBuffer();
    HostTable table = contigTable.getTable();
    return from(table, buffer);
  }

  // the returned `ColumnarBatch` warps a list of `RapidsHostColumnVectorFromBuffer` not `RapidsHostColumnVector`
  // `RapidsHostColumnVectorFromBuffer` contains a reference to the underlying `HostMemoryBuffer`
  public static ColumnarBatch from(HostTable table, HostMemoryBuffer buffer) {
    long rows = table.getRowCount();
    if (rows != (int) rows) {
      throw new IllegalStateException("Cannot support a batch larger that MAX INT rows");
    }
    int numColumns = table.getNumberOfColumns();
    RapidsHostColumnVector[] columns = new RapidsHostColumnVector[numColumns];
    try {
      for (int i = 0; i < numColumns; ++i) {
        HostColumnVector v = table.getColumn(i);
        DataType type = GpuColumnVector.getSparkType(v.getType());
        columns[i] = new RapidsHostColumnVectorFromBuffer(type, v.incRefCount(), buffer);
      }
      return new ColumnarBatch(columns, (int) rows);
    } catch (Exception e) {
      for (RapidsHostColumnVector v : columns) {
        if (v != null) {
          v.close();
        }
      }
      throw e;
    }
  }

  private RapidsHostColumnVectorFromBuffer(DataType type, HostColumnVector cudfColumn, HostMemoryBuffer buffer) {
    super(type, cudfColumn);
    this.buffer = buffer;
  }

  // copy the underlying `HostMemoryBuffer` to device，and return a `DeviceMemoryBuffer` object
  // NOTE: since all the `RapidsHostColumnVectorFromBuffer`s in `HostTable` share the same underlying `HostMemoryBuffer`, 
  //       so you should only call this method on any one of the `RapidsHostColumnVectorFromBuffer`
  public DeviceMemoryBuffer copyTheUnderlyingBufferToDevice() {
    DeviceMemoryBuffer devBuffer = DeviceMemoryBuffer.allocate(buffer.getLength());
    devBuffer.copyFromHostBuffer(buffer);
    return devBuffer;
  }

  /**
   * Get the underlying contiguous buffer, shared between columns of the original `ContiguouHostTable`
   * @return contiguous (data, validity and offsets) host memory buffer
   */
  public HostMemoryBuffer getBuffer() {
    return buffer;
  }
}
