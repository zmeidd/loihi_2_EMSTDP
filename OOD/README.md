#### Things you should change.
* **SAVED_MODEL_PATH** in  **oe_disac.py**
* **DATA_PATH** and **DATA_LABEL** in  **oe_disac.py**, the shape can be (N,32,32,1) or (N,64,64,3) or (N,32,32,3). better (N,32,32,1) or (N,64,64,3).
* **INDIST_LABEL** = [0,1,2,3,4] in  **oe_disac.py**, we choose 5 classes as the in distribution classes, and the other 5 classes are the outlier classes.
* Number of training samples, number of testing OOD samples, and the FPR percentile parameters can be changed in ood_detection in ood_detection in **oe_disac.py**.
#### You can run the OOD detection using the saved model.
```
net = Net().cuda()
net.load_state_dict(torch.load(SAVED_MODEL_PATH))
out_score = get_ood_scores(net,out_loader)
```
Then compare the out score to the threshold. The Details can be found in ood_example.ipynb.<br />
The labels can be assigned differently, so you can apply your owns. 
