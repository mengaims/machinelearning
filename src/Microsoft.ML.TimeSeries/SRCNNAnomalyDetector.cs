﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.TimeSeries;

[assembly: LoadableClass(SrCnnAnomalyDetector.Summary, typeof(IDataTransform), typeof(SrCnnAnomalyDetector), typeof(SrCnnAnomalyDetector.Options), typeof(SignatureDataTransform),
    SrCnnAnomalyDetector.UserName, SrCnnAnomalyDetector.LoaderSignature, SrCnnAnomalyDetector.ShortName)]

[assembly: LoadableClass(SrCnnAnomalyDetector.Summary, typeof(IDataTransform), typeof(SrCnnAnomalyDetector), null, typeof(SignatureLoadDataTransform),
    SrCnnAnomalyDetector.UserName, SrCnnAnomalyDetector.LoaderSignature)]

[assembly: LoadableClass(SrCnnAnomalyDetector.Summary, typeof(SrCnnAnomalyDetector), null, typeof(SignatureLoadModel),
    SrCnnAnomalyDetector.UserName, SrCnnAnomalyDetector.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(SrCnnAnomalyDetector), null, typeof(SignatureLoadRowMapper),
   SrCnnAnomalyDetector.UserName, SrCnnAnomalyDetector.LoaderSignature)]

namespace Microsoft.ML.Transforms.TimeSeries
{
    public sealed class SrCnnAnomalyDetector : SrCnnAnomalyDetectionBase, IStatefulTransformer
    {
        internal const string Summary = "This transform detects the anomalies in a time-series using SRCNN.";
        internal const string LoaderSignature = "SrCnnAnomalyDetector";
        internal const string UserName = "SrCnn Anomaly Detection";
        internal const string ShortName = "srcnn";

        internal sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "The name of the source column.", ShortName = "src",
                SortOrder = 1, Purpose = SpecialPurpose.ColumnName)]
            public string Source;

            [Argument(ArgumentType.Required, HelpText = "The name of the new column.",
                SortOrder = 2)]
            public string Name;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The size of the sliding window for computing spectral residual", ShortName = "wnd",
                SortOrder = 101)]
            public int WindowSize = 24;

            [Argument(ArgumentType.Required, HelpText = "The number of points to the back of training window.",
                ShortName = "backwnd", SortOrder = 102)]
            public int BackAddWindowSize = 5;

            [Argument(ArgumentType.Required, HelpText = "The number of pervious points used in prediction.",
                ShortName = "aheadwnd", SortOrder = 103)]
            public int LookaheadWindowSize = 5;

            [Argument(ArgumentType.Required, HelpText = "The size of sliding window to generate a saliency map for the series.",
                ShortName = "avgwnd", SortOrder = 104)]
            public int AvergingWindowSize = 3;

            [Argument(ArgumentType.Required, HelpText = "The size of sliding window to calculate the anomaly score for each data point.",
                ShortName = "jdgwnd", SortOrder = 105)]
            public int JudgementWindowSize = 21;

            [Argument(ArgumentType.Required, HelpText = "The threshold to determine anomaly, score larger than the threshold is considered as anomaly.",
                ShortName = "thre", SortOrder = 106)]
            public double Threshold = 0.3;
        }

        private sealed class SrCnnArgument : SrCnnArgumentBase
        {
            public SrCnnArgument(Options options)
            {
                Source = options.Source;
                Name = options.Name;
                WindowSize = options.WindowSize;
                InitialWindowSize = 0;
                BackAddWindowSize = options.BackAddWindowSize;
                LookaheadWindowSize = options.LookaheadWindowSize;
                AvergingWindowSize = options.AvergingWindowSize;
                JudgementWindowSize = options.JudgementWindowSize;
                Threshold = options.Threshold;
            }

            public SrCnnArgument(SrCnnAnomalyDetector transform)
            {
                Source = transform.InternalTransform.InputColumnName;
                Name = transform.InternalTransform.OutputColumnName;
                WindowSize = transform.InternalTransform.WindowSize;
                InitialWindowSize = 0;
                BackAddWindowSize = transform.InternalTransform.BackAddWindowSize;
                LookaheadWindowSize = transform.InternalTransform.LookaheadWindowSize;
                AvergingWindowSize = transform.InternalTransform.AvergingWindowSize;
                JudgementWindowSize = transform.InternalTransform.JudgementWindowSize;
                Threshold = transform.InternalTransform.AlertThreshold;
            }
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "SRCNTRNS",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(SrCnnAnomalyDetector).Assembly.FullName);
        }

        private static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));

            return new SrCnnAnomalyDetector(env, options).MakeDataTransform(input);
        }

        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            env.CheckValue(input, nameof(input));

            return new SrCnnAnomalyDetector(env, ctx).MakeDataTransform(input);
        }

        private static SrCnnAnomalyDetector Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new SrCnnAnomalyDetector(env, ctx);
        }

        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        IStatefulTransformer IStatefulTransformer.Clone()
        {
            var clone = (SrCnnAnomalyDetector)MemberwiseClone();
            clone.InternalTransform.StateRef = (SrCnnAnomalyDetectionBaseCore.State)clone.InternalTransform.StateRef.Clone();
            clone.InternalTransform.StateRef.InitState(clone.InternalTransform, InternalTransform.Host);
            return clone;
        }

        internal SrCnnAnomalyDetector(IHostEnvironment env, Options options)
            : base(new SrCnnArgument(options), LoaderSignature, env)
        {
        }

        internal SrCnnAnomalyDetector(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, ctx, LoaderSignature)
        {
        }

        private SrCnnAnomalyDetector(IHostEnvironment env, SrCnnAnomalyDetector transform)
           : base(new SrCnnArgument(transform), LoaderSignature, env)
        {
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            InternalTransform.Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>
            base.SaveModel(ctx);
        }
    }

    /// <summary>
    /// Detect anomalies in time series using Spectral Residual
    /// </summary>
    public sealed class SrCnnAnomalyEstimator : TrivialEstimator<SrCnnAnomalyDetector>
    {
        /// <param name="env">Host environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="windowSize">The size of the sliding window for computing spectral residual.</param>
        /// <param name="backAddWindowSize">The size of the sliding window for computing spectral residual.</param>
        /// <param name="lookaheadWindowSize">The number of pervious points used in prediction.</param>
        /// <param name="averagingWindowSize">The size of sliding window to generate a saliency map for the series.</param>
        /// <param name="judgementWindowSize">The size of sliding window to calculate the anomaly score for each data point.</param>
        /// <param name="threshold">The threshold to determine anomaly, score larger than the threshold is considered as anomaly.</param>
        /// <param name="inputColumnName">Name of column to transform. The column data must be <see cref="System.Single"/>.</param>
        internal SrCnnAnomalyEstimator(IHostEnvironment env,
            string outputColumnName,
            int windowSize,
            int backAddWindowSize,
            int lookaheadWindowSize,
            int averagingWindowSize,
            int judgementWindowSize,
            double threshold = 0.3,
            string inputColumnName = null)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(SrCnnAnomalyEstimator)),
                  new SrCnnAnomalyDetector(env, new SrCnnAnomalyDetector.Options
                  {
                      Source = inputColumnName ?? outputColumnName,
                      Name = outputColumnName,
                      WindowSize = windowSize,
                      BackAddWindowSize = backAddWindowSize,
                      LookaheadWindowSize = lookaheadWindowSize,
                      AvergingWindowSize = averagingWindowSize,
                      JudgementWindowSize = judgementWindowSize,
                      Threshold = threshold
                  }))
        {
        }

        internal SrCnnAnomalyEstimator(IHostEnvironment env, SrCnnAnomalyDetector.Options options)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(SrCnnAnomalyEstimator)), new SrCnnAnomalyDetector(env, options))
        {
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            if (!inputSchema.TryFindColumn(Transformer.InternalTransform.InputColumnName, out var col))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", Transformer.InternalTransform.InputColumnName);
            if (col.ItemType != NumberDataViewType.Single)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", Transformer.InternalTransform.InputColumnName, "Single", col.GetTypeString());

            var metadata = new List<SchemaShape.Column>() {
                new SchemaShape.Column(AnnotationUtils.Kinds.SlotNames, SchemaShape.Column.VectorKind.Vector, TextDataViewType.Instance, false)
            };
            var resultDic = inputSchema.ToDictionary(x => x.Name);
            resultDic[Transformer.InternalTransform.OutputColumnName] = new SchemaShape.Column(
                Transformer.InternalTransform.OutputColumnName, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Double, false, new SchemaShape(metadata));

            return new SchemaShape(resultDic.Values);
        }

    }
}
