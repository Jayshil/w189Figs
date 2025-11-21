
clearvars;

%system parameters
radius_ratio = 0.07159^2;      %(Rp/Rs)^2
rs_a = (1.0/4.607)^2;          %(Rs/a)^2
rp_a = rs_a .* radius_ratio;   %(Rp/a)^2


%Observational data
cheops_data = 87.9;
cheops_error1 = 4.3;
cheops_error2 = 4.3;
cheops_bound = [0.3750, 0.969];
cheops_center = (cheops_bound(2) - cheops_bound(1))*0.5 + cheops_bound(1);

tess_data = 203.43;
tess_error1 = 16.19;
tess_error2 = 16.27;
tess_bound = [0.5850, 1.0560];
tess_center = (tess_bound(2) - tess_bound(1))*0.5 + tess_bound(1);


%Read in the calculated emission fluxes and geometric albedos
%for the two different cases
planet_data = readmatrix("m02_0667_secondary_eclipse.dat");
mu = planet_data(:,1);  %Wavelength
em = planet_data(:,2);  %Emission flux
ag = planet_data(:,3);  %Geometric albedo


planet_data = readmatrix("m02_060_secondary_eclipse.dat");
mu = planet_data(:,1);
em_02 = planet_data(:,2);
ag_02 = planet_data(:,3);


%Read in the stellar spectrum and convert its units
stellar_data = dlmread("stellar_model.flx");

mu_star = stellar_data(:,1) .* 0.0001; %wavelength from Ang to micron

const_c = 2.99792458e10; %speed of light in cm/s

star_spectrum = stellar_data(:,2) .* const_c ./ (mu_star * 1e-4).^2; %stellar spectrum in erg s-1 cm-2 cm-1
star_spectrum = star_spectrum .* 0.001 .* 0.0001 .* 4 .* pi;         %from erg s-1 cm-2 cm-1 tp W m-2 micron-1, corrected for spherical star


%Interpolate stellar spectrum to the wavelengths of the computed model
star = interp1(mu_star, star_spectrum, mu);


%Compute the reflected flux of the seconary eclipse spectrum
refl = ag .* rs_a .* star;
refl_02 = ag_02 .* rs_a .* star;

%Compute the eclipse depths in ppm
ed = em ./ star .* radius_ratio .* 1e6 + ag .* rp_a .* 1e6;
ed_02 = em_02 ./ star .* radius_ratio .* 1e6 + ag_02 .* rp_a .* 1e6;


%Read in the filter transmission curves of TESS and CHEOPS
tess_orig = dlmread("TESS_bandpass.dat",'',6,0);
cheops_orig = dlmread("CHEOPS_bandpass.dat",'',6,0);

%Interpolate them to the wavelength grid
tess = interp1(tess_orig(:,1), tess_orig(:,2), mu, 'linear', 0);
cheops = interp1(cheops_orig(:,1), cheops_orig(:,2), mu, 'linear', 0);

%Process the stellar spectrum through the filter
star_cheops = trapz(mu, star.*cheops.*mu)./trapz(mu, cheops.*mu);
star_tess = trapz(mu, star.*tess.*mu)./trapz(mu, tess.*mu);

%Process the emitted planet flux through the filters
em_cheops = trapz(mu, em.*cheops.*mu)./trapz(mu, cheops.*mu);
em_tess = trapz(mu, em.*tess.*mu)./trapz(mu, tess.*mu);

em_cheops_02 = trapz(mu, em_02.*cheops.*mu)./trapz(mu, cheops.*mu);
em_tess_02 = trapz(mu, em_02.*tess.*mu)./trapz(mu, tess.*mu);

%Process the reflected flux through the filters
refl_cheops = trapz(mu, refl.*cheops.*mu)./trapz(mu, cheops.*mu);
refl_tess = trapz(mu, refl.*tess.*mu)./trapz(mu, tess.*mu);

refl_cheops_02 = trapz(mu, refl_02.*cheops.*mu)./trapz(mu, cheops.*mu);
refl_tess_02 = trapz(mu, refl_02.*tess.*mu)./trapz(mu, tess.*mu);

%Compute the bandpass-integrated geometric albedos
ag_cheops = refl_cheops./star_cheops ./ rs_a;
ag_tess = refl_tess./star_tess ./ rs_a;

ag_cheops_02 = refl_cheops_02./star_cheops ./ rs_a;
ag_tess_02 = refl_tess_02./star_tess ./ rs_a;


%Compute the bandpass-integrated eclipse depths
ed_cheops = em_cheops./star_cheops .* radius_ratio .* 1e6 + ag_cheops .* rp_a .* 1e6;
ed_tess = em_tess./star_tess .* radius_ratio .* 1e6 + ag_tess .* rp_a .* 1e6;

ed_cheops_02 = em_cheops_02./star_cheops .* radius_ratio .* 1e6 + ag_cheops_02 .* rp_a .* 1e6;
ed_tess_02 = em_tess_02./star_tess .* radius_ratio .* 1e6 + ag_tess_02 .* rp_a .* 1e6;


%And now we plot everything...
colors = [0    0.4470    0.7410;
          0.8500    0.3250    0.0980;
          0.9290    0.6940    0.1250;
          0.4940    0.1840    0.5560;
          0.4660    0.6740    0.1880;
          0.3010    0.7450    0.9330;
          0.6350    0.0780    0.1840];


h1=figure;

set(h1, 'Position', [200 500 700 450]);

p = plot(mu, ed, 'Linewidth', 0.5);


hold on;

p = plot(mu, ed_02, 'Linewidth', 0.5);


errorbar(cheops_center, cheops_data, cheops_error1, cheops_error2, cheops_center - cheops_bound(1), cheops_bound(2) - cheops_center,'ko','MarkerEdgeColor','k','MarkerFaceColor','k', 'CapSize', 0, 'MarkerSize',4, 'Linewidth', 2);
errorbar(tess_center, tess_data, tess_error1, tess_error2, tess_center - tess_bound(1), tess_bound(2) - tess_center,'ko','MarkerEdgeColor','k','MarkerFaceColor','k', 'CapSize', 0, 'MarkerSize',4, 'Linewidth', 2);

scatter(cheops_center, ed_cheops, 50, 'ks','MarkerEdgeColor','k','MarkerFaceColor', colors(1,:), 'LineWidth',1.5);
scatter(tess_center, ed_tess, 50, 'ks','MarkerEdgeColor','k','MarkerFaceColor', colors(1,:), 'LineWidth',1.5);

scatter(cheops_center, ed_cheops_02, 50, 'ks','MarkerEdgeColor','k','MarkerFaceColor', colors(2,:), 'LineWidth',1.5);
scatter(tess_center, ed_tess_02, 50, 'ks','MarkerEdgeColor','k','MarkerFaceColor',  colors(2,:), 'LineWidth',1.5);


xlim([-inf 1.5]);
ylim([-inf 820]);


leg = legend("$\epsilon$ = 0", "$\epsilon$ = 0.16", 'location', 'southeast', 'Interpreter', 'latex');
legend('boxoff');

set(gca,'XMinorTick','on','YMinorTick','on')
set(gca, 'FontSize', 16);

set(gca,'TickLabelInterpreter','latex');

xlabel("Wavelength ($\mu$m)", 'Interpreter', 'latex', 'FontSize', 16);
ylabel("Secondary eclipse depth (ppm)", 'Interpreter', 'latex', 'FontSize', 16);

