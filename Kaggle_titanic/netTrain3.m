


net = feedforwardnet([10,8,8,6]);
net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
net.outputs{2}.processFcns = {'removeconstantrows','mapminmax'};

net.performFcn = 'mse';

[net,tr] = train(net,train_data',train_label');

plotconfusion(train_label',net(train_data'));