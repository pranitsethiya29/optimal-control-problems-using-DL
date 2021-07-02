function output = sigmoid_dd(input)
       output = sigmoid(input) .* (1 - sigmoid(input)).* (1 - (2*sigmoid(input)));
end
