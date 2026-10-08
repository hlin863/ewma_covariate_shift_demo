function [X, featureNames, observationIndex, label] = load_park_features(source, csvPath)
%LOAD_PARK_FEATURES Read shared Python-exported CSV or a built-in demonstration.
% Compatible with scripts/export_park_matlab_features.py outputs.
% source = 'demo' or 'csv'. Local BCI GDF processing remains in Python.
if nargin < 1 || isempty(source), source = 'demo'; end
if nargin < 2, csvPath = ''; end
switch lower(char(source))
    case 'demo'
        % Deterministic, piecewise mean shift at boundary 100 (not publication data).
        rng(19,'twister');
        n = 200;
        X = [6+0.3*randn(n,1), 9+0.3*randn(n,1), 3+0.15*randn(n,1)];
        X(101:end,1) = X(101:end,1)+3;
        X(101:end,2) = X(101:end,2)+1.5;
        featureNames = {'mu_power_uv2','beta_power_uv2','rms_uv'};
        observationIndex = (0:n-1)';
        label = 'Illustrative synthetic mean change: split at 100';
    case 'csv'
        if isempty(csvPath) || ~isfile(csvPath)
            error('load_park_features:MissingCSV','A readable CSV file path is required.');
        end
        tab = readtable(csvPath);
        names = tab.Properties.VariableNames;
        timeCol = strcmp(names, 'time');
        featureNames = names(~timeCol);
        if numel(featureNames) < 2
            error('load_park_features:Columns','CSV requires at least two feature columns.');
        end
        if any(timeCol)
            observationIndex = double(tab{:,timeCol});
        else
            observationIndex = (0:height(tab)-1)';
        end
        if any(~isfinite(observationIndex)) || any(diff(observationIndex)<=0)
            error('load_park_features:Time','Time must be finite and strictly increasing.');
        end
        X = double(tab{:,~timeCol});
        if any(~isfinite(X(:))) || size(X,1) < 8
            error('load_park_features:Data','Finite features and at least eight observations required.');
        end
        label = ['CSV: ', char(csvPath)];
    otherwise
        error('load_park_features:Source','Use source ''demo'' or ''csv''.');
end
end
