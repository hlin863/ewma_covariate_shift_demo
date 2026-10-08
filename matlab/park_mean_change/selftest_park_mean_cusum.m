function selftest_park_mean_cusum()
%SELFTEST_PARK_MEAN_CUSUM Numerical and failure-case checks without toolboxes.
X = [repmat([2,3],40,1); repmat([7,8],40,1)];
r = park_mean_cusum(X,8);
assert(r.bMax==40 && r.bAvg==40,'Expected step change at boundary 40.');
assert(numel(r.positions)==65,'Candidate position count mismatch.');
assert(size(r.byChannel,2)==2,'Feature axis mismatch.');
% Compare original weighted sums to simplified weighted difference of means.
for j = [8,20,40,60]
  expected = sqrt(j*(80-j)/80) .* ...
      abs(mean(X(1:j,:),1)-mean(X(j+1:end,:),1)) ./ mean(X,1);
  ix = find(r.positions==j,1);
  assert(max(abs(r.byChannel(ix,:)-expected))<1e-10,'Equation mismatch.');
end
try
    park_mean_cusum([ones(80,1),zeros(80,1)],8);
    error('selftest:ExpectedFailure','Zero feature mean was accepted.');
catch ME
    if strcmp(ME.identifier,'selftest:ExpectedFailure'), rethrow(ME); end
    assert(strcmp(ME.identifier,'park_mean_cusum:Means'));
end
disp('Park mean-CUSUM MATLAB self-test passed.');
end
