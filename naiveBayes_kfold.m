% Import data
Z = csvread('clean_manageable_data_creditcard.csv', 1); % 1 = second row (no headers)

% Define data
Y = Z(:,end); % output (response)
X = Z(:,1:6); % model inputs (predictors)
classNames = {'pay','default'}; % INTENTAR que funcione, no funciona

% Define the right distribution for each variable/feature
distribution_types = string();
distribution_types(1) = 'kernel'; % best result according to 'OptimizeHyperparameters', is a kernel, box, Width = 15695
distribution_types(2:4) = 'mvmn';
distribution_types(5) = 'normal'; % comentar la grafica de edad para justificar esto
distribution_types(6) = 'mvmn';
distribution_types = cellstr(distribution_types);

% Define categorical variables/features
categorical_variables = [2 3 4 6];

% Prior probabilities
prior = [0.78 0.22]; % the prior should be between [0.5 0.5](1) and [0.78 0.22](2) 
                   % due to 1/nº responses(1) and due to dataset distribution (2)

% Cost matrix (missclassification) IMPORTANT, explain in the report.
miss_matrix = [0 1.5;3.5 0]; % the optimal result usually got unbalanced regarding default payments
                           % therefore, we apply this matrix to classify better

% Naive Bayes Classifier
naive = fitcnb(X, Y,'DistributionNames',distribution_types,'CategoricalPredictors',categorical_variables,...
    'Kfold',10,'Kernel','box','Width',15695,'Support','positive','Prior',prior,'Cost',miss_matrix);
%naive.DistributionParameters
%naive.DistributionParameters{1,2}
naive

%naive_loan_ammount = fitcnb(X(:,1), Y,'OptimizeHyperparameters','all'); %averiguar el kernel de esta distribucion 
    %(graficamente no la reconocemos), caso contrario a la variables "age" que tiene una normal clarisima)
%naive_loan_ammount

validation_error_kfold = kfoldLoss(naive, 'mode', 'individual'); % con esto puedo comparar MODELOS, poner perdida para cada "naive" y puedes ver las diferencias.
validation_error_kfold

[min_validation_loss, P] = min(validation_error_kfold); %Selecciona el modelo con MENOR error, es magia! perdida vs perdida_2
[min_validation_loss, P]

Y_predict = predict(naive.Trained{P}, X); % predicted values (vector)
c_matrix = confusionmat(Y,Y_predict); % Confusion matrix
c_matrix
HeatMap(c_matrix); % Confusion matrix representation

model_accurancy = sum(Y_predict == Y)/length(Y) * 100; % accurancy level
model_accurancy

% Plots
gscatter(X(:,5),X(:,1),Y); % X axis (age), Y axis (Loan amount) 
                           % Blue (default), Red (well paid, non default)

% FALTA INCLUIR LA ROC, la grafica para justificar todo, mirar ML video
% matlab

% Conclusion: 
% Cuanto mayor es el accurancy, mayor es la desigualdad en la distribucion de probabilidad (mirar c_matrix)
% Por tanto, es necesario la matriz de misclasificacion, aunque perdamos
% accurancy, el modelo es mas correcto (EXPLICAR ESTO BIEN). 
% miss_matrix = [0 1.5;3.5 0], es la matrix perfecta.
% La matrix esta bien enfocada, porque NOS INTERESA saber QUIEN va a pagar,
% osea nos interesa una prediccion de 0 alta, no de 1 (default).

% (Lo que he hecho de jugar con los parametros, seguro que se puede
% automatizar, mirar como). (DistributionParameters ???)

% la prior buena, y estable es, prior = [0.78 0.22];; en conjunto con la
% miss_matrix. Encima, es la prior REAL de los datos.

% al sistema le cuesta calcular el default porque hay pocos casos en
% nuestra muestra, es decir, es por culpa de nuestros datos, porque tenemos
% pocos casos posibles en los que eso pueda ocurrir.
