function output = template_matching(im1, im2)
    im1 = imresize(im1, [320 320]);
    im2 = imresize(im2, [320 320]);
    output = zeros(size(im1, 1), size(im1, 2));
    for i = 5 : size(im1, 1) - 5
        for j = 5 : size(im1, 2) - 5
            wnd = im1(i - 4: i + 4, j - 4 : j + 4, 1 : 3);
            diff = 1000000000000000000;
            idx = -1;
            for k = 5 : size(im1, 2) - 5
                cmp = im2(i - 4: i + 4, k - 4: k + 4, 1 : 3);
                tmp = (wnd - cmp);
                tmp = tmp .* tmp;
                tmp = sum(sum(sum(tmp)));
                if tmp < diff
                    diff = tmp;
                    idx = k;
                end
            end
            output(i, j) = abs(j - idx);
        end
    end
end