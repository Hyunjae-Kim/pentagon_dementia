OC_NN.ipynb -> OC_NN_model.py : R값을 같이 update하도록
    
OC_NN_quantileR.ipynb -> OC_NN_model2.py : 
논문에서 언급한대로 nu-quantile 하게 R값을 설정하도록 한것.

결과는 OC_NN_model2.py를 토대로 OC_NN_ROC_sweep.py/ROC_sweep.sh 로 뽑음.

ROC_check.ipynb를 통해 ROC plot, AUC 계산.
AUC는 하나의 hidden에 대해서 nu바꿔가며 학습했을때 optimal하게 학습된 경우에 대해서
종합하여 계산. (코드참고)

# OC_NN_ROC_sweep2.py는 각 모델(파라미터)에 따라 threshold(r값)을 sweep해서 제일 좋은 
# AUC취하는 방식. 위에 방법보다 AUC가 낮음.

MMSE와 상관관계는 ROC커브중 가장 왼쪽 위에 해당하는 모델(nu, iteration)으로 얻은
output value들과 MMSE와의 상관관계를 계산.