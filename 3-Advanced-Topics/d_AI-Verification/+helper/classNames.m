function names = classNames(idx)
%classNames Return ACAS Xu advisory class names
%   names = classNames returns all 5 advisory class names as a categorical
%   array: Clear-of-Conflict, Weak Left, Weak Right, Strong Left, Strong Right.
%   names = classNames(idx) returns the class name(s) at the given index.

nameList = [ ...
    "Clear-of-Conflict"; ...
    "Weak Left"; ...
    "Weak Right"; ...
    "Strong Left"; ...
    "Strong Right"];
allNames = categorical(nameList, nameList);

if nargin == 0
    names = allNames;
else
    names = allNames(idx);
end
end
