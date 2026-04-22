function [loss, gradients] = modelLoss(net, X, T)
if nargout > 1
    Y = net.forward(X);
else
    Y = net.predict(X);
end

loss = crossentropy(Y,T);
seqLength = size(X, 3);
loss = loss / seqLength; % So loss doesn't scale with sequence length

if nargout > 1
    gradients = dlgradient(loss, net.Learnables);
end
end