
from astroquery.mast import Mast
from astroquery.mast import Observations

keplerObs = Observations.query_criteria(target_name='kplr011446443', obs_collection='Kepler')
keplerProds = Observations.get_product_list(keplerObs[1])
yourProd = Observations.filter_products(keplerProds, extension='kplr011446443-2009131110544_slc.fits', 
                                        mrp_only=False)
Observations.download_products(yourProd, mrp_only = False, cache = False) 
