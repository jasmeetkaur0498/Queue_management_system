function data = preprocess_data(filename, savePath)
% PREPROCESS_DATA  Load, clean, compute wait/service times, save cleaned data.
if nargin < 2, savePath = 'results'; end
if ~exist(savePath,'dir'), mkdir(savePath); end

fprintf('Reading %s ...\n', filename);
opts = detectImportOptions(filename,'NumHeaderLines',0);
T = readtable(filename, opts);

cols = T.Properties.VariableNames;
fprintf('Columns detected: %s\n', strjoin(cols,', '));

% Try parse datetimes if present
try
    if ismember('arrival_time', cols) && ~isdatetime(T.arrival_time)
        T.arrival_time = tryParseDatetime(T.arrival_time);
    end
    if ismember('start_time', cols) && ~isdatetime(T.start_time)
        T.start_time = tryParseDatetime(T.start_time);
    end
    if ismember('finish_time', cols) && ~isdatetime(T.finish_time)
        T.finish_time = tryParseDatetime(T.finish_time);
    end
catch
end

n = height(T);
% compute wait_time, service_time
T.wait_time = nan(n,1); T.service_time = nan(n,1);
if ismember('arrival_time', cols) && isdatetime(T.arrival_time) && ismember('start_time', cols) && isdatetime(T.start_time)
    T.wait_time = minutes(T.start_time - T.arrival_time);
end
if ismember('start_time', cols) && isdatetime(T.start_time) && ismember('finish_time', cols) && isdatetime(T.finish_time)
    T.service_time = minutes(T.finish_time - T.start_time);
end

% fallback numeric columns or fill zeros
if all(isnan(T.wait_time))
    if ismember('wait_time', cols) && ~isdatetime(T.wait_time)
        % preserve existing numeric wait_time if any
    else
        T.wait_time = zeros(n,1);
    end
end
T.wait_time(T.wait_time < 0) = NaN;
T.service_time(T.service_time < 0) = NaN;

% Drop rows missing wait_time (optional)
validMask = ~isnan(T.wait_time);
if sum(~validMask) > 0
    fprintf('Dropping %d rows with missing wait_time\n', sum(~validMask));
    T = T(validMask,:);
end

% baseline
if ismember('wait_time', T.Properties.VariableNames)
    baseline_avg_wait = mean(T.wait_time(~isnan(T.wait_time)));
else
    baseline_avg_wait = NaN;
end

fprintf('Preprocessing completed. Records retained: %d. Baseline avg wait = %.3f minutes\n', height(T), baseline_avg_wait);

outmat = fullfile(savePath,'preprocessed_queue_data.mat');
outcsv = fullfile(savePath,'preprocessed_queue_data.csv');
save(outmat,'T','baseline_avg_wait');
writetable(T,outcsv);

data = T;
data.Properties.UserData.baseline_avg_wait = baseline_avg_wait;
end

%% helpers
function dt = tryParseDatetime(col)
    if isdatetime(col), dt = col; return; end
    colStr = string(col);
    fmts = {'dd-MM-yyyy HH.mm','yyyy-MM-dd HH:mm:ss.SSS','yyyy-MM-dd HH:mm:ss','dd-MM-yyyy HH:mm','MM/dd/yyyy HH:mm:ss','dd/MM/yyyy HH:mm'};
    for k=1:length(fmts)
        try
            dt = datetime(colStr,'InputFormat',fmts{k});
            return;
        catch
        end
    end
    try
        dt = datetime(colStr);
    catch
        error('Failed to parse datetime column. Provide consistent time format.');
    end
end
