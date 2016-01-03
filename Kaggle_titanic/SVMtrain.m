
ind = randperm(891);

cutpoint = 800;

model = svmtrain(train_label(ind(1:cutpoint)) , train(ind(1:cutpoint), :));
[ptrain] = svmpredict(train_label(ind(cutpoint+1:891)) , train(ind(cutpoint+1:891), :) , model);