<pre>
org.deeplearning4j.eval.Evaluation eval = new org.deeplearning4j.eval.Evaluation(3);
org.nd4j.linalg.api.ndarray.INDArray output = model.output(testData.getFeatureMatrix());
eval.eval(testData.getLabels(), output);
log.info(eval.stats());
</pre>
