/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationState;

import lombok.Getter;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.common.io.stream.Writeable;

import java.io.IOException;

/**
 * A wrapper class for serializing and deserializing float[] arrays using Writeable.
 */
@Getter
class FloatArrayWrapper implements Writeable {

    private final float[] array;

    public FloatArrayWrapper(float[] array) {
        this.array = array;
    }

    public FloatArrayWrapper(StreamInput in) throws IOException {
        this.array = in.readFloatArray();
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeFloatArray(array);
    }
}
