from train import create_model_train
import numpy as np
from sklearn.preprocessing import LabelEncoder
from data_prepare import get_data

#get test data
x_test,y_test = get_data(mode = "test")

#process the data
x_test = np.expand_dims(x_test, axis=3)
en_t = LabelEncoder()
en_t.fit(y_test)
y_test_encoded = en_t.transform(y_test)

#prediction
model = create_model_train()
prediction = model.predict(x_test)


#print 10 sample test results
print("10 sample test results")
test_truth = [ y_test_encoded[i] for i in range(10) ]
test_result = [ np.argmax(prediction[i]) for i in range(10) ]
testdata = {'test_truth': test_truth, 'test_result': test_result}
result_table = pd.DataFrame(testdata)
print(result_table)
