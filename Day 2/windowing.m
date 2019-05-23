function output = windowing(img, centre, width)
    for i = 1 : size(img, 1)
        for j = 1 : size(img, 2)
            for k = 1 : size(img, 3)
                if img(i, j, k) > (centre + width)
                    img(i, j, k) = centre + width;
                end
                if img(i, j, k) < (centre - width)
                    img(i, j, k) = centre - width;
                end
            end
        end
    end
    output = mat2gray(img);
end