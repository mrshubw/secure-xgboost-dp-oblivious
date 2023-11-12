import numpy as np
import securexgboost as xgb
import unittest
import os
import json
import pytest
import locale
from sklearn.datasets import dump_svmlight_file
from config import sym_key_file, priv_key_file, cert_file

username = "user1"
HOME_DIR = os.path.dirname(os.path.realpath(__file__)) + "/../../"

temp_name = HOME_DIR + "demo/data/temp_file.txt"
temp_enc_name = HOME_DIR + "demo/data/temp_file.txt.enc"

dpath = HOME_DIR + 'demo/data/'
dtrain = xgb.DMatrix({username: dpath + 'agaricus.txt.train.enc'})
dtest = xgb.DMatrix({username: dpath + 'agaricus.txt.test.enc'})

rng = np.random.RandomState(1994)


class TestModels(unittest.TestCase):
    def test_glm(self):
        param = {'verbosity': 0, 'objective': 'binary:logistic',
                 'booster': 'gblinear', 'alpha': 0.0001, 'lambda': 1,
                 'nthread': 1}
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = 4
        bst = xgb.train(param, dtrain, num_round, watchlist)
        assert isinstance(bst, xgb.core.Booster)
        preds = bst.predict(dtest)[0]
        #TODO(rishabh): implement get_label()
        """
        labels = dtest.get_label()
        err = sum(1 for i in range(len(preds))
                  if int(preds[i] > 0.5) != labels[i]) / float(len(preds))
        assert err < 0.2
        """

    def test_dart(self):
        dtrain = xgb.DMatrix({username: dpath + 'agaricus.txt.train.enc'})
        dtest = xgb.DMatrix({username: dpath + 'agaricus.txt.test.enc'})
        param = {'max_depth': 5, 'objective': 'binary:logistic',
                 'eval_metric': 'logloss', 'booster': 'dart', 'verbosity': 1}
        # specify validations set to watch performance
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = 2
        bst = xgb.train(param, dtrain, num_round, watchlist)
        # this is prediction
        preds = bst.predict(dtest, ntree_limit=num_round)[0]
        #TODO(rishabh): implement get_label()
        """
        labels = dtest.get_label()
        err = sum(1 for i in range(len(preds))
                  if int(preds[i] > 0.5) != labels[i]) / float(len(preds))
        # error must be smaller than 10%
        assert err < 0.1
        """

        #TODO(rishabh): implement save_binary()
        """
        # save dmatrix into binary buffer
        dtest.save_binary('dtest.buffer')
        model_path = 'xgb.model.dart'
        # save model
        bst.save_model(model_path)
        # load model and data in
        bst2 = xgb.Booster(params=param, model_file='xgb.model.dart')
        dtest2 = xgb.DMatrix('dtest.buffer')
        preds2 = bst2.predict(dtest2, ntree_limit=num_round)[0]
        # assert they are the same
        assert np.sum(np.abs(preds2 - preds)) == 0
        """

        def my_logloss(preds, dtrain):
            return
            #TODO(rishabh): implement get_label()
            """
            labels = dtrain.get_label()
            return 'logloss', np.sum(
                np.log(np.where(labels, preds, 1 - preds)))
            """

        # check whether custom evaluation metrics work
        #TODO: implement feval (allow definition of a loss function?)
        """
        bst = xgb.train(param, dtrain, num_round, watchlist,
                        feval=my_logloss)
        preds3 = bst.predict(dtest, ntree_limit=num_round)[0]
        assert all(preds3 == preds)
        """

        #TODO(rishabh): implement get_label()
        """
        # check whether sample_type and normalize_type work
        num_round = 50
        param['verbosity'] = 0
        param['learning_rate'] = 0.1
        param['rate_drop'] = 0.1
        preds_list = []
        for p in [[p0, p1] for p0 in ['uniform', 'weighted']
                  for p1 in ['tree', 'forest']]:
            param['sample_type'] = p[0]
            param['normalize_type'] = p[1]
            bst = xgb.train(param, dtrain, num_round, watchlist)
            preds = bst.predict(dtest, ntree_limit=num_round)[0]
            err = sum(1 for i in range(len(preds))
                      if int(preds[i] > 0.5) != labels[i]) / float(len(preds))
            assert err < 0.1
            preds_list.append(preds)

        for ii in range(len(preds_list)):
            for jj in range(ii + 1, len(preds_list)):
                assert np.sum(np.abs(preds_list[ii] - preds_list[jj])) > 0
        os.remove(model_path)
        """

    def run_eta_decay(self, tree_method):
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = 4

        # learning_rates as a list
        # init eta with 0 to check whether learning_rates work
        param = {'max_depth': 2, 'eta': 0, 'verbosity': 0,
                 'objective': 'binary:logistic', 'tree_method': tree_method}
        evals_result = {}
        #TODO(rishabh): implement callbacks
        """
        bst = xgb.train(param, dtrain, num_round, watchlist,
                        callbacks=[xgb.callback.reset_learning_rate([
                            0.8, 0.7, 0.6, 0.5
                        ])],
                        evals_result=evals_result)
        eval_errors_0 = list(map(float, evals_result['eval']['error']))
        assert isinstance(bst, xgb.core.Booster)
        # validation error should decrease, if eta > 0
        assert eval_errors_0[0] > eval_errors_0[-1]
        """

        # init learning_rate with 0 to check whether learning_rates work
        #TODO(rishabh): implement callbacks
        """
        param = {'max_depth': 2, 'learning_rate': 0, 'verbosity': 0,
                 'objective': 'binary:logistic', 'tree_method': tree_method}
        evals_result = {}
        bst = xgb.train(param, dtrain, num_round, watchlist,
                        callbacks=[xgb.callback.reset_learning_rate(
                            [0.8, 0.7, 0.6, 0.5])],
                        evals_result=evals_result)
        eval_errors_1 = list(map(float, evals_result['eval']['error']))
        assert isinstance(bst, xgb.core.Booster)
        # validation error should decrease, if learning_rate > 0
        assert eval_errors_1[0] > eval_errors_1[-1]
        """

        # check if learning_rates override default value of eta/learning_rate
        #TODO(rishabh): implement callbacks
        """
        param = {
            'max_depth': 2, 'verbosity': 0, 'objective': 'binary:logistic',
            'tree_method': tree_method
        }
        evals_result = {}
        bst = xgb.train(param, dtrain, num_round, watchlist,
                        callbacks=[xgb.callback.reset_learning_rate(
                            [0, 0, 0, 0]
                        )],
                        evals_result=evals_result)
        eval_errors_2 = list(map(float, evals_result['eval']['error']))
        assert isinstance(bst, xgb.core.Booster)
        # validation error should not decrease, if eta/learning_rate = 0
        assert eval_errors_2[0] == eval_errors_2[-1]
        """

        # learning_rates as a customized decay function
        def eta_decay(ithround, num_boost_round):
            return num_boost_round / (ithround + 1)

        #TODO(rishabh): implement callbacks
        """
        evals_result = {}
        bst = xgb.train(param, dtrain, num_round, watchlist,
                        callbacks=[
                            xgb.callback.reset_learning_rate(eta_decay)
                        ],
                        evals_result=evals_result)
        eval_errors_3 = list(map(float, evals_result['eval']['error']))

        assert isinstance(bst, xgb.core.Booster)

        assert eval_errors_3[0] == eval_errors_2[0]

        for i in range(1, len(eval_errors_0)):
            assert eval_errors_3[i] != eval_errors_2[i]
        """

    def test_eta_decay_hist(self):
        self.run_eta_decay('hist')

    def test_eta_decay_approx(self):
        self.run_eta_decay('approx')

    def test_eta_decay_exact(self):
        self.run_eta_decay('exact')

    def test_boost_from_prediction(self):
        # Re-construct dtrain here to avoid modification
        margined = xgb.DMatrix({username: dpath + 'agaricus.txt.train.enc'})
        bst = xgb.train({'tree_method': 'hist'}, margined, 1)
        predt_0 = bst.predict(margined, output_margin=True)
        #TODO(rishabh): implement set_base_margin()
        """
        margined.set_base_margin(predt_0)
        bst = xgb.train({'tree_method': 'hist'}, margined, 1)
        predt_1 = bst.predict(margined)[0]

        assert np.any(np.abs(predt_1 - predt_0) > 1e-6)

        bst = xgb.train({'tree_method': 'hist'}, dtrain, 2)
        predt_2 = bst.predict(dtrain)[0]
        assert np.all(np.abs(predt_2 - predt_1) < 1e-6)
        """

    def test_custom_objective(self):
        param = {'max_depth': 2, 'eta': 1, 'verbosity': 0}
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = 2

        def logregobj(preds, dtrain):
            labels = dtrain.get_label()
            preds = 1.0 / (1.0 + np.exp(-preds))
            grad = preds - labels
            hess = preds * (1.0 - preds)
            return grad, hess

        def evalerror(preds, dtrain):
            labels = dtrain.get_label()
            return 'error', float(sum(labels != (preds > 0.0))) / len(labels)

        # test custom_objective in training
        #TODO(rishabh): support custom objective and loss in train()
        """
        bst = xgb.train(param, dtrain, num_round, watchlist, logregobj, evalerror)
        assert isinstance(bst, xgb.core.Booster)
        preds = bst.predict(dtest)[0]
        labels = dtest.get_label()
        err = sum(1 for i in range(len(preds))
                  if int(preds[i] > 0.5) != labels[i]) / float(len(preds))
        assert err < 0.1
        """

        #TODO(rishabh): implement cv()
        """
        # test custom_objective in cross-validation
        xgb.cv(param, dtrain, num_round, nfold=5, seed=0,
               obj=logregobj, feval=evalerror)
        """

        # test maximize parameter
        def neg_evalerror(preds, dtrain):
            return
            #TODO(rishabh): implement get_label()
            """
            labels = dtrain.get_label()
            return 'error', float(sum(labels == (preds > 0.0))) / len(labels)
            """

        #TODO(rishabh): support custom objective and loss in train()
        """
        bst2 = xgb.train(param, dtrain, num_round, watchlist, logregobj, neg_evalerror, maximize=True)
        preds2 = bst2.predict(dtest)[0]
        err2 = sum(1 for i in range(len(preds2))
                   if int(preds2[i] > 0.5) != labels[i]) / float(len(preds2))
        assert err == err2
        """

    def test_multi_eval_metric(self):
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        param = {'max_depth': 2, 'eta': 0.2, 'verbosity': 1,
                 'objective': 'binary:logistic'}
        param['eval_metric'] = ["auc", "logloss", 'error']
        evals_result = {}
        #TODO(rishabh): support evals_result
        """
        bst = xgb.train(param, dtrain, 4, watchlist, evals_result=evals_result)
        assert isinstance(bst, xgb.core.Booster)
        assert len(evals_result['eval']) == 3
        assert set(evals_result['eval'].keys()) == {'auc', 'error', 'logloss'}
        """

    def test_fpreproc(self):
        param = {'max_depth': 2, 'eta': 1, 'verbosity': 0,
                 'objective': 'binary:logistic'}
        num_round = 2

        def fpreproc(dtrain, dtest, param):
            return
            #TODO(rishabh): implement get_label()
            """
            label = dtrain.get_label()
            ratio = float(np.sum(label == 0)) / np.sum(label == 1)
            param['scale_pos_weight'] = ratio
            return (dtrain, dtest, param)
            """

        #TODO(rishabh): implement cv()
        """
        xgb.cv(param, dtrain, num_round, nfold=5,
               metrics={'auc'}, seed=0, fpreproc=fpreproc)
        """

    def test_show_stdv(self):
        param = {'max_depth': 2, 'eta': 1, 'verbosity': 0,
                 'objective': 'binary:logistic'}
        num_round = 2
        #TODO(rishabh): implement cv()
        """
        xgb.cv(param, dtrain, num_round, nfold=5,
               metrics={'error'}, seed=0, show_stdv=False)
        """

    def test_feature_names_validation(self):
        X = np.random.random((10, 3))
        y = np.random.randint(2, size=(10,))

        dump_svmlight_file(X, y, temp_name) 
        xgb.encrypt_file(temp_name, temp_enc_name, sym_key_file)
 
        dm1 = xgb.DMatrix({username: temp_enc_name})
        dm2 = xgb.DMatrix({username: temp_enc_name}, feature_names=("a", "b", "c"))

        bst = xgb.train([], dm1)
        bst.predict(dm1)  # success
        self.assertRaises(ValueError, bst.predict, dm2)
        bst.predict(dm1)  # success

        bst = xgb.train([], dm2)
        bst.predict(dm2)  # success
        self.assertRaises(ValueError, bst.predict, dm1)
        bst.predict(dm2)  # success

