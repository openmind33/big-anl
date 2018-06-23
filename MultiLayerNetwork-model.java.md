
org.deeplearning4j.nn.multilayer.MultiLayerNetwork model 
= new org.deeplearning4j.nn.multilayer.MultiLayerNetwork(conf);
model.init();
model.setListeners(new ScoreIterationListener(100));
for(int i=0; i<1000; i++ ) {
    model.fit(trainingData);
}



