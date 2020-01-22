import bert_classifier
from utils import config

config.USE_GPU = False

cm = bert_classifier.ClassificationModel(gpu=False, seed=0)
if config.load_frompretrain == True:
    cm.load_model(config.model_state_path, config.model_config_path)
else:
    cm.new_model()

actual_to_save, predictions_to_save = cm.create_test_predictions("./pred.csv")

from sklearn.metrics import f1_score

print("BERT classifier, F1 score is {}".format(f1_score(actual_to_save,predictions_to_save,average='macro')))

if __name__ == '__main__':
    print('running')
    config.data_path = 'data/'
    config.USE_GPU = False
    config.save_path = 'save/'
