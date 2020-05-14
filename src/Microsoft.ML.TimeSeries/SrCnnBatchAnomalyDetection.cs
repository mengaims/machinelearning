// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.Data.DataView;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Numeric;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.TimeSeries
{
    /// <summary>
    /// The detect modes of SrCnn models.
    /// </summary>
    public enum SrCnnDetectMode
    {
        /// <summary>
        /// In this mode, output (IsAnomaly, RawScore, Mag).
        /// </summary>
        AnomalyOnly = 0,

        /// <summary>
        /// In this mode, output (IsAnomaly, AnomalyScore, Mag, ExpectedValue, BoundaryUnit, UpperBoundary, LowerBoundary).
        /// </summary>
        AnomalyAndMargin = 1,

        /// <summary>
        /// In this mode, output (IsAnomaly, RawScore, Mag, ExpectedValue).
        /// </summary>
        AnomalyAndExpectedValue = 2
    }

    // TODO: SrCnn
    internal sealed class SrCnnBatchAnomalyDetector : BatchDataViewMapperBase<float, SrCnnBatchAnomalyDetector.Batch>
    {
        private readonly int _batchSize;
        private const int _minBatchSize = 12;
        private readonly string _inputColumnName;

        private class Bindings : ColumnBindingsBase
        {
            private readonly DataViewType _outputColumnType;
            private readonly int _inputColumnIndex;

            public Bindings(DataViewSchema input, string inputColumnName, string outputColumnName, DataViewType outputColumnType)
                : base(input, true, outputColumnName)
            {
                _outputColumnType = outputColumnType;
                _inputColumnIndex = Input[inputColumnName].Index;
            }

            protected override DataViewType GetColumnTypeCore(int iinfo)
            {
                Contracts.Check(iinfo == 0);
                return _outputColumnType;
            }

            // Get a predicate for the input columns.
            public Func<int, bool> GetDependencies(Func<int, bool> predicate)
            {
                Contracts.AssertValue(predicate);

                var active = new bool[Input.Count];
                for (int col = 0; col < ColumnCount; col++)
                {
                    if (!predicate(col))
                        continue;

                    bool isSrc;
                    int index = MapColumnIndex(out isSrc, col);
                    if (isSrc)
                        active[index] = true;
                    else
                        active[_inputColumnIndex] = true;
                }

                return col => 0 <= col && col < active.Length && active[col];
            }
        }

        public SrCnnBatchAnomalyDetector(IHostEnvironment env, IDataView input, string inputColumnName, string outputColumnName, double threshold, int batchSize, double sensitivity, SrCnnDetectMode detectMode)
            : base(env, "SrCnnBatchAnomalyDetector", input, new Bindings(input.Schema, inputColumnName, outputColumnName, NumberDataViewType.Single))
        {
            Contracts.CheckParam(batchSize >= _minBatchSize, nameof(batchSize), "batch size is too small");
            _batchSize = batchSize;
            _inputColumnName = inputColumnName;
        }

        protected override Delegate[] CreateGetters(DataViewRowCursor input, Batch currentBatch, bool[] active)
        {
            if (!SchemaBindings.AnyNewColumnsActive(x => active[x]))
                return new Delegate[1];
            return new[] { currentBatch.CreateGetter(input, _inputColumnName) };
        }

        protected override Batch InitializeBatch(DataViewRowCursor input) => new Batch(_batchSize);

        protected override Func<bool> GetIsNewBatchDelegate(DataViewRowCursor input)
        {
            return () => input.Position % _batchSize == 0;
        }

        protected override Func<bool> GetLastInBatchDelegate(DataViewRowCursor input)
        {
            return () => (input.Position + 1) % _batchSize == 0;
        }

        protected override ValueGetter<float> GetLookAheadGetter(DataViewRowCursor input)
        {
            return input.GetGetter<float>(input.Schema[_inputColumnName]);
        }

        protected override Func<int, bool> GetSchemaBindingDependencies(Func<int, bool> predicate)
        {
            return (SchemaBindings as Bindings).GetDependencies(predicate);
        }

        protected override void ProcessExample(Batch currentBatch, float currentInput)
        {
            currentBatch.AddValue(currentInput);
        }

        protected override void ProcessBatch(Batch currentBatch)
        {
            currentBatch.Process();
            currentBatch.Reset();
        }

        public sealed class Batch
        {
            private List<float> _previousBatch;
            private List<float> _batch;
            private float _cursor;
            private readonly int _batchSize;

            public Batch(int batchSize)
            {
                _batchSize = batchSize;
                _previousBatch = new List<float>(batchSize);
                _batch = new List<float>(batchSize);
            }

            public void AddValue(float value)
            {
                _batch.Add(value);
            }

            public int Count => _batch.Count;

            public void Process()
            {
                // TODO: replace with SrCnn
                _cursor = VectorUtils.NormSquared(new ReadOnlySpan<float>(_batch.ToArray()));
                if (_batch.Count < _batchSize)
                {
                    _cursor += VectorUtils.NormSquared(new ReadOnlySpan<float>(
                        _previousBatch.GetRange(_batch.Count, _batchSize - _batch.Count).ToArray()));
                }
            }

            public void Reset()
            {
                var tempBatch = _previousBatch;
                _previousBatch = _batch;
                _batch = tempBatch;
                _batch.Clear();
            }

            public ValueGetter<float> CreateGetter(DataViewRowCursor input, string inputCol)
            {
                ValueGetter<float> srcGetter = input.GetGetter<float>(input.Schema[inputCol]);
                ValueGetter<float> getter =
                    (ref float dst) =>
                    {
                        float src = default;
                        srcGetter(ref src);
                        dst = src * _cursor;
                    };
                return getter;
            }
        }
    }
}
