ar = zeros(50, 2);
for i = 1 : 50
    ar(i, 1) = floor(i / 10) + rand(1) * .5;
    ar(i, 2) = floor(i / 10) + rand(1) * .5;
end
scatter(ar(:, 1), ar(:, 2));