import numpy as np
from src.adaptation import DecisionTreeClassifier, KNNClassifier, LinearSVMClassifier
from src.adaptation.decision_support import explain_decision_tree, explain_knn, explain_linear_svm, explain_pwknn, serialise_decision_tree
from src.adaptation.pseudo_labelling import PWKNNPseudoLabeler

def _data():
    x=np.array([[-2.,-.5],[-1.5,-.2],[-1.,.1],[1.,-.1],[1.5,.2],[2.,.5]])
    return x,np.array([0,0,0,1,1,1])

def test_interpretable_classifiers_share_feature_contract():
    x,y=_data()
    for model in (DecisionTreeClassifier(max_depth=3,min_samples_leaf=1).fit(x,y),KNNClassifier(n_neighbors=3).fit(x,y),LinearSVMClassifier().fit(x,y)):
        assert model.predict(np.array([[-1.2,0.],[1.2,0.]])).tolist()==[0,1]
        assert model.training_size==6

def test_decision_evidence_exposes_distinct_reasoning_modes():
    x,y=_data(); names=("f1","f2"); sample=np.array([1.4,.1])
    tree=DecisionTreeClassifier(max_depth=3,min_samples_leaf=1).fit(x,y); knn=KNNClassifier(n_neighbors=3).fit(x,y); svm=LinearSVMClassifier().fit(x,y); pw=PWKNNPseudoLabeler(n_neighbors=3,rbf_sigma=1.)
    te=explain_decision_tree(tree,sample,names); ke=explain_knn(knn,sample); pe=explain_pwknn(pw,reference_features=x,reference_labels=y,sample=sample); se=explain_linear_svm(svm,sample,names); structure=serialise_decision_tree(tree,names)
    assert te["prediction"]==1 and te["path"][-1]["kind"]=="leaf"
    assert ke["prediction"]==1 and len(ke["neighbours"])==3
    assert pe["prediction"]==1 and .5<=pe["confidence"]<=1.
    assert se["prediction"]==1 and se["top_contributions"]
    assert structure["node_id"]==0
