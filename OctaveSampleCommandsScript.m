% Maths

5+6
3-2
5*8
1/2
2^6

% Logical Operations

1 == 2 %This is comment
1 ~= 2 % 1 not equal to 2 -> its true hence value is 1
1 && 0 % AND Operation
1 || 0 % OR Operation
xor (1,0)

% Other Basic Operations

PS1('Octave>> '); % To change prompt
a = 3; % Semicolon suppresses output
b = 'hi'
c = (3>=1)
a=pi
disp(a);
disp(sprintf('Two Decimal : %0.2f',a)); % Display decimal upto 2 decimal points
disp(sprintf('Six Decimal : %0.6f',a)); % Display decimal upto 6 decimal points
format long
format short


% Matrices

A = [1 2; 3 4; 5 6] % 3X2 Matrices
Vrow = [1 2 3]  % Row vector
Vcol = [1;2;3]  % Column vector
V = 1:0.1:2  % Vector starts at 1 and increments by 0.1 and goes upto 2
V1 = 1:6
A = ones (2,3)  % Generates 2X3 matric of all ones
C = 2*ones(2,3)  % Generates 2X3 matric of all ones multiplied by 2
D = zeros(1,3)  % Generates 1X3 matric of all zeros
E = rand(3,4)  % Generates 3X4 matric of random numbers
F = randn(2,3) % Generates 2X3 matric of Gaussian distribution with mean zero and variance or Std Dev equal to one
G = -6+sqrt(10)*(randn(1,10000))
hist(G)  % Generates histogram of G
hist(G,50) % Generates histogram of G with bin of 50
eye(4) % Generate identity matrix i.e. diagnol values of 1
help eye % This is to get documentation help on specific command

% Moving around Data

size (A) % Returns size of a matric
size (A,2) % Returns size of row of the matric
length (Vcol) % Returns longer dimension of array, primarily used for vectors
pwd % Displays current directory
cd 'c:\' % changes directory
load('filename') % Loads file in Octave environment
who % Shows list of variables in memory
whos % Gives detailed view of variables in memory
clear A % removes A from memory
save OctaveSampleCommandTest.m A  % Saves variable A in file
save Octave.text A -ascii  % Saves variable A in human readable file
clear  % Deletes all variables in enviornment
A = [1 2;3 4;5 6]
A(3,2)  % Prints element in 3rd row and 2nd column of Matric A
A(2,:)  % Fetch everything in second row, here : means every element of that row/columns
A([1 3],:)  % Get everything from 1st and 3rd row
A(:,2) = [10; 11; 12]  % Replace elements of second column by 10,11, 12
A = [A, [100; 101; 102]];  % Append another column vector to the right
A(:)  % Creates one column vector for all element in the matrix
A = [1 2;3 4;5 6]
B = [100 200;300 400;500 600]
C = [A B]  % Concatenates matric A and B
C = [A; B]  % Concatenates matric B at the bottom of A

% Computational Operations on Data

A = [1 2;3 4;5 6]
B = [11 12; 13 14; 15 16]
C = [1 1; 2 2]
A*C   % Multiplication of Matrics
A .*B   % Multiple each element by respective element of other matrics
A .^2   % Element wise squaring of A
V = [1;2;3]
1./V   % Element wise reciprocal of Vector V
log(V)   % Log of Vector V
exp(V)   % Exponential of Vector V
abs(V)   % Element wise absolute value of V
-V   % Minus each element of V
V + ones(length(V),1)   % Increment each element of V by 1
V + 1   % Increments each element of V by 1. It is same as above command
A'   % Transpose of Matrics A
a = [1 15 30 0.5]
max(A)   % Returns row with max value of A
val = max(a)   % Returns max of a
[val,ind] = max(a)   % Returns max value and its index of a
a < 3   % Element wise comparison with 3 and creating matrix with 1 for true and 0 for false
find(a<3)   % This will find element index of a that are less than 3
B = magic(3)   % All rows, columns and diagonals sum up to same value
[r,c] = find (B >= 7)   % Find r row index and c column index where B element is > 7
B(2,3)  % Prints element of B at row 2 and column 3
sum(B)  % Sums all column elements of B and create 3X1 matric
prod(B)  % Multiply all column elements of B and create 3X1 matric
floor(a)  % Returns round numbers
ceil(a)   % Returns nearest integer
max(rand(3), rand(3))
C = [8 1 6; 3 5 7; 4 9 2]
max(C, [], 1)    % Calculates column wise maximum in C (mXn) and creates 1Xn matric with max values for every column
max(C, [], 2)    % Calculates row wise maximum in C (mXn) and creates mX1 matric with max values for every row
max(max(C))   % Calculates max of C
max(C(:))     % Calculates max of C by converting C into 1 column vector
C.*eye(3)     % Wipe out all other elements from Matric C and leave only diagonal elements
sum(sum(C.*eye(3))   % Calculates sum of diagonal elements of matric C
sum(sum(C.*flipud(eye(3))))   % Calculates sum of opposite diagonal elements of matric C; flipud is called flip updown
pinv(C)   % Calculates inverse of matric C

% Plotting Data

t = [0:0.01:0.98]  % Row matric with values from 0 to 0.98 with difference of 0.01 between each of them
y1 = sin(2*pi*4*t)
y2 = cos(2*pi*4*t)
plot(t,y1)    % plots sin graph
hold on;      % Holds on to existing figure and draw new ones on top of its
plot(t,y2)    % plots cos graph on top of sin graph
plot(t,y2,'r')   % 'r' represents red color for the graph
xlabel('Time')   % Label X axis
ylabel('Value')   % Label Y axis
legend('sin','cos')   % Label lines with sin and cos
title('my plot')   % This is title of graph
print -dpng 'myplot.png'   % This will save graph in a file
help plot   % To see other file formats in which graph can be saved
close    % To close figure window
figure(1); plot(t,y1)   % Create different figure window for the graph
figure(2); plot(t,y2)   % Create different figure window for the graph
subplot(1,2,1);    % Divides plot a 1X2 grid and starts to plot first element
plot(t,y1)
subplot(1,2,2);    % Divides plot a 1X2 grid and starts to plot second element
plot(t,y2)
axis([0.5 1 -1 1])   % Formats axis - horizontal axis from 0.5 to 1 and vertical axis from -1 to 1
clf   % Clears a figure
A = magic(5)
imagesc(A)   % Grid of colors where color represents values
imagesc(A), colorbar, colormap gray   % Plots the grid of color on grey scale and also draws the scale; we are using comma chaining function calls

% Control Statements: If, For, While

v=zeros(10,1)
for i = 1:10,    % For loop statement
  v(i) = 2^i;
end;

indices = 1:10
for i = indices;
  disp(i);
end;

i = 1;
while i <= 10;   % While loop statement
  v(i) = 100;
  i = i+1;
end;

i = 1;
while true;      % Break statement
  v(i) = 999;
  i = i+1;
  if (i == 10);
    break;
  end;
end;

if v(1) == 999   % If Else statement
    disp('The value is 999')
  elseif v(2) == 100
    disp('The value is 100')
  else
    disp('The value is neither 999 or 100')
end;

% Defining Functions - create a file with name of function and end in m

addpath('D:\Kshitij-Google-Drive\Learning\Machine-Learning\Coursera-Harvard-MachineLearning\machine-learning-ex1\ex1')   % This will add location to Octave search path
function y = SquareThisNumber(x)   % Defines a function that returns value y and takes argument x as input; Should be in file name SquareThisNumber.m
  y = x^2
  
function [y1, y2] = SquareAndCubeThisNumber(x)   % Defines function with x as input and y1, y2 as output; Should be in file name SquareAndCubeThisNumber.m
  y1 = x^2;
  y2 = x^3;

function J = costFunctionJ(X, y, theta)
  % X is the design matrix containing our training example; X = [1 1; 1 2; 1 3]
  % y is the class label; y = [1; 2; 3]
  % theta = [0;1]
  
  m = size(X,1)    % Number of training examples
  predictions = X* theta   % prediction of hypothesis on all m examples
  sqrErrors = (predictions-y).^2   % Squared errors
  J = 1/(2*m) * sum(sqrErrors);

% Vectorization

prediction = 0.0;   % Unvectorized implementation 
for j = 1:n+1,
  prediction = prediction + theta(j) + x(j)
end;

prediction = theta'*x   % Vectorized implementation


  
  










