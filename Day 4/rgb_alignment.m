function res = rgb_alignment(img)
    ht = floor(size(img, 1) / 3);
    [b, rect] = imcrop(img, [1 1 size(img, 2) ht]);
    [g, rect] = imcrop(img, [1 (1 * ht) size(img, 2) ht]);
    [r, rect] = imcrop(img, [1 (2 * ht) size(img, 2) ht]);
    c = normxcorr2(g, b);
    [ypeak, xpeak] = find(c==max(c(:)));
    yoffSetg = ypeak-size(g,1);
    xoffSetg = xpeak-size(g,2);
    c = normxcorr2(r, b);
    [ypeak, xpeak] = find(c==max(c(:)));
    yoffSetr = ypeak-size(r,1);
    xoffSetr = xpeak-size(r,2);
    if xoffSetg < 0
        green = g(:, abs(xoffSetg) + 1 : end);
        green = padarray(green, [0 abs(xoffSetg)], 0, 'post');
    end
    if xoffSetg >= 0
        green = g(:, xoffSetg + 1 : end);
        green = padarray(green, [0 abs(xoffSetg)], 0, 'pre');
    end
    if yoffSetg < 0
        green = green(abs(yoffSetg) + 1 : end, :);
        green = padarray(green, [abs(yoffSetg) 0], 0, 'post');
    end
    if yoffSetg >= 0
        green = green(abs(yoffSetg) + 1 : end, :);
        green = padarray(green, [abs(yoffSetg) 0], 0, 'pre');
    end
    
    if xoffSetr < 0
        red = r(:, abs(xoffSetr) + 1 : end);
        red = padarray(red, [0 abs(xoffSetr)], 0, 'post');
    end
    if xoffSetr >= 0
        red = r(:, xoffSetr + 1 : end);
        red = padarray(red, [0 abs(xoffSetr)], 0, 'pre');
    end
    if yoffSetr < 0
        red = red(abs(yoffSetr) + 1 : end, :);
        red = padarray(red, [abs(yoffSetr) 0], 0, 'post');
    end
    if yoffSetr >= 0
        red = red(abs(yoffSetr) + 1 : end, :);
        red = padarray(red, [abs(yoffSetr) 0], 0, 'pre');
    end
    res(:, :, 1) = red;
    res(:, :, 2) = green;
    res(:, :, 3) = b;
end