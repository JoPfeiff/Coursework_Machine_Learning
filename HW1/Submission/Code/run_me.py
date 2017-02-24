# run_me.py module

import four_and_five_hyperparameters as fourfive
import three_default_classifier_accuracy as three
import six_get_creative as six

# Assuming you are running run_me.py from the Submission/Code directory, otherwise the path variable will be different for you


#three.question_3()
#three.data_set_for_bargraph()


fourfive.optimize_knn_email()
fourfive.acc_default_knn_email()
fourfive.optimize_knn_occupancy()
fourfive.optimize_knn_USPS()
fourfive.optimize_decision_tree_Email()
fourfive.optimize_decision_tree_occupancy()
fourfive.optimize_decision_tree_USPS()
fourfive.optimize_AdaBoost_Email()
fourfive.optimize_AdaBoost_occupancy()
fourfive.optimize_AdaBoost_USPS()


six.adaBoost_final_optimization_email()
six.adaBoost_final_optimization_occupancy()
six.adaBoost_final_optimization_USPS()

