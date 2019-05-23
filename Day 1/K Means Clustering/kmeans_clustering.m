function means = kmeans_clustering(k, X)
    means = [];       % Contains the mean values of k clusters
    N = size(X, 1);   % Number of data points
    dim = size(X, 2); % Number of dimensions of each training example
    % Randomly initialize the mean values of k clusters
    for i = 1 : k
        tmp = [];
        for j = 1 : dim
            minimum = ceil(min(X(:, j)));
            maximum = floor(max(X(:, j)));
            tmp = [tmp randi([minimum maximum])];
        end
        means = [means; tmp];
    end
    % Initialise the labels of each data point as 0
    labels = zeros(N, 1);
    iterations = 0;
    % Run the algorithm till convergence
    while true
        iterations = iterations + 1;
        for sample = 1 : N
            % For each sample find the cluster whose mean is nearest.
            idx = -1; res = 10000000000;
            % Looping through each of the clusters
            for j = 1 : k
                val = 0;
                % Sum error for dim dimensions.
                for i = 1 : dim
                    val = val + (((X(sample, i) - means(j, i)) * (X(sample, i) - means(j, i))));
                end
                val = sqrt(val);
                % Update if distance is better than previous distance
                if(val < res)
                    res = val;
                    idx = j;
                end
            end
            % Assign apt. label to the data point
            labels(sample, 1) = idx;
        end
        % Compute new means
        new_means = means;
        for j = 1 : k
            S = zeros(1, dim);
            cnt = 0;
            for i = 1 : N
                for l = 1 : dim
                   if(labels(i, 1) == j)
                       cnt = cnt + 1;
                       S(1, l) = S(1, l) + X(i, l);
                   end
                end
            end
            if(cnt ~= 0)
                S = S ./ (cnt / dim);
                new_means(j, :) = S;
            end
        end
        % Check for convergence
        chk = (new_means == means);
        br = 1;
        for i = 1 : k
            for j = 1 : dim
                if(chk(i, j) == 0)
                    br = 0;
                    break;
                end
            end
        end
        % If converged, break out of loop
        if br == 1
            break;
        end
        % Otherwise update means to new_means
        means = new_means;
    end
    % Visualization works only for 2d data when cluster size is less than equal to 5 %
%     hold on;
%     for i = 1 : N
%         c = [];
%         if (labels(i, 1) == 1)
%             c = [1 0 0];
%         end
%         if (labels(i, 1) == 2)
%             c = [0 1 0];
%         end
%         if (labels(i, 1) == 3)
%             c = [0 0 1];
%         end
%         if (labels(i, 1) == 4)
%             c = [1 1 0];
%         end
%         if (labels(i, 1) == 5)
%             c = [1 0 1];
%         end
%         scatter(X(i, 1), X(i, 2), 40 , 'MarkerEdgeColor',c, 'MarkerFaceColor', c);
%     end
%     hold off;
end