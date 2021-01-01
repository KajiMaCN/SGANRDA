clc
clear
load mnist_uint8;

train_x = double(train_x(1:60000,:)) / 255;

train_y = double(ones(size(train_x,1),1));
% normalize
train_x = mapminmax(train_x, 0, 1);

rand('state',0)

test_x = normrnd(0,1,[60000,100]); 
test_x = mapminmax(test_x, 0, 1);

test_y = double(zeros(size(test_x,1),1));
test_y_rel = double(ones(size(test_x,1),1));

nn_G_t = nnsetup([100 784]);
nn_G_t.activation_function = 'sigm';
nn_G_t.output = 'sigm';

nn_D = nnsetup([784 100 1]);
nn_D.weightPenaltyL2 = 1e-4;  %  L2 weight decay
nn.dropoutFraction = 0.5;   %  Dropout fraction 
nn.learningRate = 0.01;                %  Sigm require a lower learning rate
nn_D.activation_function = 'sigm';
nn_D.output = 'sigm';
% nn_D.weightPenaltyL2 = 1e-4;  %  L2 weight decay

nn_G = nnsetup([100 784 100 1]);
nn_D.weightPenaltyL2 = 1e-4;  %  L2 weight decay
nn.dropoutFraction = 0.5;   %  Dropout fraction 
nn.learningRate = 0.01;                %  Sigm require a lower learning rate
nn_G.activation_function = 'sigm';
nn_G.output = 'sigm';
% nn_G.weightPenaltyL2 = 1e-4;  %  L2 weight decay

opts.numepochs =  1;        %  Number of full sweeps through data
opts.batchsize = 100;       %  Take a mean gradient step over this many samples

num = 1000;
tic
for each = 1:1500
    for i = 1:length(nn_G_t.W)   
        nn_G_t.W{i} = nn_G.W{i};
    end
    G_output = nn_G_out(nn_G_t, test_x);
    
    index = randperm(60000); 
    train_data_D = [train_x(index(1:num),:);G_output(index(1:num),:)];
    train_y_D = [train_y(index(1:num),:);test_y(index(1:num),:)];
    nn_D = nntrain(nn_D, train_data_D, train_y_D, opts);
    for i = 1:length(nn_D.W)  
        nn_G.W{length(nn_G.W)-i+1} = nn_D.W{length(nn_D.W)-i+1};
    end
    nn_G = nntrain(nn_G, test_x(index(1:num),:), test_y_rel(index(1:num),:), opts);
end
toc
for i = 1:length(nn_G_t.W)
    nn_G_t.W{i} = nn_G.W{i};
end
fin_output = nn_G_out(nn_G_t, test_x);
