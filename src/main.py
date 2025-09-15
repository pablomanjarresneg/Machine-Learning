from deps import *

hdul = fits.open("data\lightcurve.fits")
data = hdul[1].data  # Table with the light curve

# Extract time and flux
time = data['TIME']
flux = data['PDCSAP_FLUX']

# Plot
plt.figure(figsize=(10,5))
plt.plot(time, flux, ".", markersize=2)
plt.xlabel("Time (days)")
plt.ylabel("Corrected Flux (PDCSAP_FLUX)")
plt.title("TESS Light Curve")
plt.show()