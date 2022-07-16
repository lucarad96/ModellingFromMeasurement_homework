function [outputArg1,outputArg2] = complete_matrix_from_samples(inputArg1,inputArg2)

% NON UNA FUNZIONE VERA E PROPRIA, LO FACCIAMO PIù PER SALVARE LA LOGICA DI
% RICOSTRUIRE UNA MATRICE INTERA (N_LEARNER,N_SNAPSHOTS) DA ALCUNE
% PREVISIONI IN SPECIFICI MOMENTI DI TEMPO.

% Rebuilding the matrix with NaN for snapshots where the values were not
% computed
bag_ex_dmd.hare_tot = NaN(n_learners,n_snapshots);
bag_ex_dmd.lynx_tot = NaN(n_learners,n_snapshots);
for ii = 1:n_snapshots
    t_curr = t(ii);
    for jj = 1:n_samples
        for zz = 1:n_learners
            if bag_ex_dmd.t(zz,jj) == t_curr
                bag_ex_dmd.hare_tot(zz,ii) = bag_ex_dmd.hare(zz,jj);
                bag_ex_dmd.lynx_tot(zz,ii) = bag_ex_dmd.lynx(zz,jj);
            end
        end
    end
end

% Computation of mean and STD
bag_ex_dmd.hare_mean = mean(bag_ex_dmd.hare_tot,"omitnan");
bag_ex_dmd.hare_std = std(bag_ex_dmd.hare_tot,"omitnan");
bag_ex_dmd.lynx_mean = mean(bag_ex_dmd.lynx_tot,"omitnan");
bag_ex_dmd.lynx_std = std(bag_ex_dmd.lynx_tot,"omitnan");

end