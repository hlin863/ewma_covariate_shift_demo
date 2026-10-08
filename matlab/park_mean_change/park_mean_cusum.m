function result = park_mean_cusum(X, minSegment)
%PARK_MEAN_CUSUM Exploratory Park et al. (2023) Section 2.1 CUSUM curves.
% Rows: ordered observations (EEG trials). Columns: positive-mean features.
% Equivalent to src/detection/stage1/park_mean_cusum.py.
% Returns candidate locations ONLY: no unimodality, significance or alarms.
if nargin < 2, minSegment = 8; end
validateattributes(X, {'numeric'}, {'2d','real','finite','nonempty'}, mfilename, 'X');
[n, p] = size(X);
if p < 2, error('park_mean_cusum:Columns', 'At least two feature columns are required.'); end
validateattributes(minSegment, {'numeric'}, {'scalar','integer','>=',2}, mfilename, 'minSegment');
if 2*minSegment >= n
    error('park_mean_cusum:Length', 'Need n > 2*minSegment.');
end
featureMeans = mean(X, 1);
if any(featureMeans <= 1e-12)
    error('park_mean_cusum:Means', 'All feature means must be positive (>1e-12). Use positive power features, not signed raw EEG.');
end
positions = (minSegment:(n-minSegment))';
prefix = [zeros(1,p); cumsum(double(X), 1)];
before = prefix(positions+1,:);
after = prefix(end,:) - before;
left = sqrt((n-positions)./(n.*positions));
right = sqrt(positions./(n.*(n-positions)));
% bsxfun is retained for compatibility with older MATLAB releases.
curves = bsxfun(@rdivide, abs(bsxfun(@times,left,before) - ...
    bsxfun(@times,right,after)), featureMeans);
maximum = max(curves, [], 2);
average = mean(curves, 2);
[~,imax] = max(maximum);
[~,iavg] = max(average);
result.positions = positions;
result.byChannel = curves; % MATLAB: [candidate positions x variables]
result.maximum = maximum;
result.average = average;
result.bMax = positions(imax);
result.bAvg = positions(iavg);
result.agreementGap = abs(result.bMax-result.bAvg);
result.featureMeans = featureMeans;
result.minSegment = minSegment;
result.nObservations = n;
result.nVariables = p;
end
