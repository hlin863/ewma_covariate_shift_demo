function fig = plot_park_mean_change(X, featureNames, positionsOrTime, result, label)
%PLOT_PARK_MEAN_CHANGE Compare input features, per-feature CUSUM, aggregates.
% Candidate positions are splits AFTER t observations (matching Python).
if nargin < 5, label = 'Multivariate feature sequence'; end
if nargin < 3 || isempty(positionsOrTime)
    positionsOrTime = (0:size(X,1)-1)';
end
if nargin < 2 || isempty(featureNames)
    featureNames = arrayfun(@(k) sprintf('Variable %d',k), ...
        1:size(X,2),'UniformOutput',false);
end
fig = figure('Name','Park Section 2.1 - Mean change exploration','Color','w');
subplot(3,1,1);
plot(positionsOrTime, X, 'LineWidth', 1);
title([label,' | ordered positive features'], 'Interpreter','none');
ylabel('Feature value'); grid on; legend(featureNames,'Location','best');
subplot(3,1,2);
plot(result.positions, result.byChannel, 'LineWidth',1.15);
hold on; xline(result.bMax,'--','max peak'); xline(result.bAvg,':','avg peak');
title('Normalised channel-wise CUSUM: nu_t^(d)');
ylabel('CUSUM score'); grid on; legend(featureNames,'Location','best');
subplot(3,1,3);
plot(result.positions,result.maximum,'LineWidth',1.7); hold on;
plot(result.positions,result.average,'LineWidth',1.7);
xline(result.bMax,'--','b_{max}'); xline(result.bAvg,':','b_{avg}');
ylabel('Aggregated CUSUM'); xlabel('Candidate split position (trial count)');
title(sprintf('b_{max}=%d | b_{avg}=%d | difference=%d (candidate peaks only)',...
    result.bMax,result.bAvg,result.agreementGap));
legend({'Maximum over variables','Average over variables'},'Location','best');
grid on;
end
