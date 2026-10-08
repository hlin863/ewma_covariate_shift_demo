function result = run_park_mean_change(source, csvPath, minSegment, outputFolder)
%RUN_PARK_MEAN_CHANGE Entry point for Section 2.1 MATLAB visualisation.
% Examples:
%   run_park_mean_change
%   run_park_mean_change('csv','outputs/matlab/park_2b_subject01_session01.csv')
% Execute from repository root after adding matlab/park_mean_change to path,
% or from the MATLAB folder containing these functions.
if nargin < 1 || isempty(source), source = 'demo'; end
if nargin < 2, csvPath = ''; end
if nargin < 3 || isempty(minSegment), minSegment = 8; end
if nargin < 4, outputFolder = ''; end
[X, names, time, label] = load_park_features(source, csvPath);
result = park_mean_cusum(X, minSegment);
plot_park_mean_change(X,names,time,result,label);
fprintf('Source: %s | observations: %d | variables: %d\n',...
    label,result.nObservations,result.nVariables);
fprintf('Candidate b_max=%d, b_avg=%d, gap=%d. No confirmed alarm.\n',...
    result.bMax,result.bAvg,result.agreementGap);
if ~isempty(outputFolder)
    if ~isfolder(outputFolder), mkdir(outputFolder); end
    outputTable = table(result.positions,result.maximum,result.average,...
        'VariableNames',{'position','maximum','average'});
    writetable(outputTable,fullfile(outputFolder,'park_mean_change_curves.csv'));
    exportgraphics(gcf,fullfile(outputFolder,'park_mean_change.png'),'Resolution',160);
end
end
