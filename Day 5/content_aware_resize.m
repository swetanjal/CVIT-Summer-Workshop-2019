function output = content_aware_resize(img, remove)
    im = rgb2gray(img);
    h = imgradient(im);
    cost = zeros(size(h, 1), size(h, 2));
    px = zeros(size(h, 1), size(h, 2));
    for i = 1 : size(h, 1)
        for j = 1 : size(h, 2)
            px(i, j) = j;
        end
    end
    columns_to_be_removed = remove;
    for iter = 1 : columns_to_be_removed
        for j = 1 : size(h, 2)
            cost(1, j) = h(i, j);
        end
        for i = 2 : size(h, 1)
            for j = 1 : size(h, 2)
                minimum = 1000000000000000000;
                if (j - 1) >= 1
                    minimum = min(minimum, cost(i - 1, j - 1));
                end
                if (j + 1) <= size(h, 2)
                    minimum = min(minimum, cost(i - 1, j + 1));
                end
                minimum = min(minimum, cost(i - 1, j));
                cost(i, j) = h(i, j) + minimum;
            end
        end
        pos = -1;
        m = 1000000000000000000;
        for j = 1 : size(h, 2)
            if cost(size(h, 1), j) <= m
                m = cost(size(h, 1), j);
                pos = j;
            end
        end
        for i = size(h, 1) : -1 : 1
            next_pos = -1;
            M = 1000000000000000000;
            if i == 1
                cost(i, :) = [cost(i, 1 : pos - 1) cost(i, pos + 1 : end) 0];
                px(i, : ) = [px(i, 1 : pos - 1) px(i, pos + 1 : end) 0];
                h(i, : ) = [h(i, 1 : pos - 1) h(i, pos + 1 : end) 0];
                break;
            end
            if (pos - 1) >= 1 && cost(i - 1, pos - 1) <= M
                M = cost(i - 1, pos - 1);
                next_pos = pos - 1;
            end
            if (pos + 1) <= size(h, 2) && cost(i - 1, pos + 1) <= M
                M = cost(i - 1, pos + 1);
                next_pos = pos + 1;
            end
            if cost(i - 1, pos) <= M
                M = cost(i - 1, pos);
                next_pos = pos;
            end
            cost(i, :) = [cost(i, 1 : pos - 1) cost(i, pos + 1 : end) 0];
            im(i, : ) = [im(i, 1 : pos - 1) im(i, pos + 1 : end) 0];
            h(i, : ) = [h(i, 1 : pos - 1) h(i, pos + 1 : end) 0];
            px(i, : ) = [px(i, 1 : pos - 1) px(i, pos + 1 : end) 0];
            pos = next_pos;
        end
        cost = cost(:, 1 : size(cost, 2) - 1);
        im = im(:, 1 : size(im, 2) - 1);
        h = h(:, 1 : size(h, 2) - 1);
        px = px(:, 1 : size(px, 2) - 1);
    end
    for i = 1 : size(px, 1)
        for j = 1 : size(px, 2)
            for k = 1 : 3
                output(i, j, k) = img(i, px(i, j), k);
            end
        end
    end
    imshow(output);
end