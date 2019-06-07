function query(img)
    img = imread(img);
    im = imcrop(img);
    imwrite(im, 'query.jpg');
end