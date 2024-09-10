from ml_module import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import datetime
import argparse

    
parser = argparse.ArgumentParser(description='Train binary classifier models.')
parser.add_argument('--models', type=str, default = 'xgboost', help='Comma-separated list of models to train (e.g., xgboost,DNN). Default is xgboost only.')
args = parser.parse_args()
model_list = args.models.split(',')

# model_list = ['xgboost', 'DNN']

# Load training data
data = pd.read_csv('train.csv')
# Data preprocessing
preprocessor = DataPreprocess(data)
X,y = preprocessor.preprocess()

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)

# Scale features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Load test.csv
inference_data = pd.read_csv('test.csv')
# Data preprocessing
inf_preprocessor = DataPreprocess(inference_data)
X_inf = inf_preprocessor.preprocess(inference = True)
X_inf = scaler.transform(X_inf)

roc_dict ={}

for model_name in model_list:
    print(f'For {model_name} :')
    # Train model
    model = Train_model(X_train, y_train, model_name = model_name)

    # Make prediction on test dataset
    y_pred, y_pred_prob = make_prediction(model, model_name, X_test)

    # Make inference on test.csv
    inference_data = pd.read_csv('test.csv')
    y_inf, y_inf_prob = make_prediction(model, model_name, X_inf)
    inference_data['Exited_Inference'] = y_inf


    # Model Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrix_df = pd.DataFrame(conf_matrix, columns = ['Predict_0', 'Predict_1'], index = ['True_0', 'True_1'])
    class_report = classification_report(y_test, y_pred, output_dict=True)
    class_report_df = pd.DataFrame(class_report).transpose()

    # Write outputs to Excel
    right_now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    with pd.ExcelWriter('Inference_{}_{}.xlsx'.format(model_name,right_now)) as writer:
        inference_data.to_excel(writer, sheet_name='result', index=False)
        class_report_df.to_excel(writer, sheet_name='pred_ability')
        conf_matrix_df.to_excel(writer, sheet_name='confusion_matrix')

    roc_dict.update({model_name:{'y_test':y_test, 'y_pred_prob':y_pred_prob}})
    print(f'{model_name} is completed')


# Plot ROC curve
plt.figure(figsize=(8, 4))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
for model_name in roc_dict:
    y_test = roc_dict[model_name]['y_test']
    y_pred_prob = roc_dict[model_name]['y_pred_prob']

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    auc_score = roc_auc_score(y_test, y_pred_prob)
    plt.plot(fpr, tpr, label=f'{model_name} (area = {auc_score:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')

# Save ROC curve to png file
plt.savefig('ROC_curve_{}.png'.format(right_now), format='png')



















        