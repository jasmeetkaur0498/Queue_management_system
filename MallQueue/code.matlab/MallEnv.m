classdef MallEnv
    properties
        arrivalIAT
        serviceDist
        arrivalRate
        numCounters = 6
        maxQueueLength = 30
        timeStepMinutes = 1
        baseline_avg_wait = 1.0
        rngSeed = 1
        verbose = false
        arrivalScale = 1.0
        serviceScale = 1.0

        state
        active
        params
    end

    methods
        function obj = MallEnv(data, params)
            if nargin < 2, params = struct(); end
            obj.params = params;
            if isfield(params,'numCounters'), obj.numCounters = params.numCounters; end
            if isfield(params,'maxQueueLength'), obj.maxQueueLength = params.maxQueueLength; end
            if isfield(params,'timeStepMinutes'), obj.timeStepMinutes = params.timeStepMinutes; end
            if isfield(params,'seed'), obj.rngSeed = params.seed; end
            if isfield(params,'verbose'), obj.verbose = params.verbose; end
            if isfield(params,'arrivalScale'), obj.arrivalScale = params.arrivalScale; end
            if isfield(params,'serviceScale'), obj.serviceScale = params.serviceScale; end

            rng(obj.rngSeed);

            obj.state = zeros(1, obj.numCounters);
            obj.active = ones(1, obj.numCounters);
            obj.arrivalIAT = 2 + 0.5*randn(100,1);
            obj.serviceDist = 7 + 2*randn(100,1);
            obj.arrivalRate = 0.5;

            if nargin>=1 && istable(data)
                try
                    if ismember('arrival_time', data.Properties.VariableNames) && isdatetime(data.arrival_time)
                        iat = minutes(diff(sort(data.arrival_time)));
                        if ~isempty(iat), obj.arrivalIAT = max(0.1, iat(:)); end
                    end
                catch
                end
                try
                    if ismember('service_time', data.Properties.VariableNames)
                        s = data.service_time; s = s(~isnan(s) & s>0);
                        if ~isempty(s), obj.serviceDist = s(:); end
                    end
                catch
                end
                try
                    if ismember('wait_time', data.Properties.VariableNames)
                        obj.baseline_avg_wait = mean(data.wait_time(~isnan(data.wait_time)));
                    end
                catch
                end
                try
                    if ismember('arrival_time', data.Properties.VariableNames) && isdatetime(data.arrival_time)
                        totalMinutes = minutes(max(data.arrival_time) - min(data.arrival_time));
                        if totalMinutes > 0
                            obj.arrivalRate = height(data) / max(1, totalMinutes);
                        end
                    else
                        muIAT = mean(obj.arrivalIAT(:));
                        obj.arrivalRate = 1 / max(0.1, muIAT);
                    end
                catch
                    obj.arrivalRate = 0.5;
                end
            end

            % apply arrivalScale/serviceScale
            obj.arrivalRate = obj.arrivalRate * obj.arrivalScale;
            obj.serviceDist = obj.serviceDist * (1/obj.serviceScale);

            % cap by num counters (safe)
            obj.arrivalRate = min(obj.arrivalRate, 5 * obj.numCounters);

            if obj.verbose
                fprintf('MallEnv constructed: counters=%d, maxQ=%d, arrivalRate=%.3f/min, meanService=%.3f min\n', ...
                    obj.numCounters, obj.maxQueueLength, obj.arrivalRate, mean(obj.serviceDist(:)));
            end
        end

        function obs = reset(obj)
            obj.state = zeros(1, obj.numCounters);
            obj.active = ones(1, obj.numCounters);
            obs = obj.observe();
        end

        function obs = observe(obj)
            avgQueue = mean(obj.state);
            obs = [obj.state, sum(obj.active), avgQueue];
        end

        function [obs, reward, done, obj] = step(obj, action)
            C = obj.numCounters;
            prioritized = -1;
            if action >= 1 && action <= C
                prioritized = action;
            elseif action > C && action <= 2*C
                idx = action - C;
                obj.active(idx) = 0;
            end

            avgService = mean(obj.serviceDist(:));
            if isempty(avgService) || ~isfinite(avgService) || avgService <= 0
                p_serv = 0.1;
            else
                p_serv = min(1.0, obj.timeStepMinutes * 1.2 / avgService); % slight boost
            end

            for c = 1:C
                if obj.active(c) == 1 && obj.state(c) > 0
                    if rand() < p_serv
                        obj.state(c) = obj.state(c) - 1;
                    end
                end
            end

            if prioritized > 0 && obj.active(prioritized) == 1 && obj.state(prioritized) > 0
                if rand() < p_serv
                    obj.state(prioritized) = max(0, obj.state(prioritized) - 1);
                end
            end

            arrRate = obj.arrivalRate;
            maxArrivalsPerMinute = 5 * obj.numCounters;
            arrRate = min(arrRate, maxArrivalsPerMinute);

            numNew = poissrnd(arrRate * obj.timeStepMinutes);
            for k = 1:numNew
                idx = randi(C);
                obj.state(idx) = min(obj.state(idx) + 1, obj.maxQueueLength);
            end

            avgWait = mean(obj.state);
            activeCount = sum(obj.active);
            obs = [obj.state, activeCount, avgWait];

            base = obj.baseline_avg_wait;
            if isempty(base) || ~isfinite(base) || base <= 0
                base = 1.0;
            end
            reward = 100 * (base - avgWait) / base;
            reward = max(-100, min(100, reward));

            done = (avgWait > 0.98 * obj.maxQueueLength);

            if obj.verbose
                fprintf('step: arrivals=%d, avgWait=%.3f, reward=%.3f\n', numNew, avgWait, reward);
            end
        end
    end
end
