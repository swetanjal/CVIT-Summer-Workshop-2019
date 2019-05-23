function tomography(img, angle_step)
    img = padarray(img, [size(img, 1), size(img, 2)]);
    mat = zeros(size(img, 1), size(img, 2));
    for i = 0 : 179
        rotated_img = imrotate(img, angle_step, 'crop', 'bilinear');
        rotated_mat = imrotate(mat, angle_step, 'crop', 'bilinear');
        rotated_mat = rotated_mat + repmat((sum(rotated_img, 1) ./ size(rotated_img, 1)), [size(rotated_img, 1) 1]);
        img = rotated_img;
        mat = rotated_mat;
    end
    imshow(mat);
end