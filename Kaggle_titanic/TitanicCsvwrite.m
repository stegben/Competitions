function h = TitanicCsvwrite(result,filename)
    survive = zeros(1,418);
    for k = 1:418
        if result(k)<0.5
            survive(k) = 0
        else
            survive(k) = 1;
        end
    end
    csvwrite(filename , )
end