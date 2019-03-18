
from classifier.classifier import predict
from sketchRnn.generateNewFill import generate

fill=generate("./temp/test.mid")
print(predict(fill))
