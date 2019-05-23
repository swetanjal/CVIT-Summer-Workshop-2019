function res = ChromaKey(fg, bg, keyColor)
    fg = double(fg); bg = double(bg);
    res = zeros(size(fg, 1), size(fg, 2), 3);
    eps = 175;
    for i = 1 : size(fg, 1)
        for j = 1 : size(fg, 2)
            score = abs(fg(i, j, 1) - keyColor(1)) * 1.0 * abs(fg(i, j, 1) - keyColor(1)) + abs(fg(i, j, 2) - keyColor(2)) *1.0 * abs(fg(i, j, 2) - keyColor(2)) + abs(fg(i, j, 3) - keyColor(3)) * 1.0 * abs(fg(i, j, 3) - keyColor(3));
            score = sqrt(1.0 * score * 1.0);
            if  score <= eps
                res(i, j, 1) = bg(i, j, 1);
                res(i, j, 2) = bg(i, j, 2);
                res(i, j, 3) = bg(i, j, 3);
            else
                res(i, j, 1) = fg(i, j, 1);
                res(i, j, 2) = fg(i, j, 2);
                res(i, j, 3) = fg(i, j, 3);
            end
        end
    end
    res = uint8(res);
    imshow(res);
end