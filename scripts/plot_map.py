import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

code = 'HOU'
if code == 'HOU':
    ax_extent = [-105, -85, 25, 35]
elif code == 'SEUS':
    ax_extent = [-90, -75, 30, 37.5]    

states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')

fig = plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent(ax_extent)
ax.add_feature(cfeature.LAND)
ax.add_feature(states_provinces, edgecolor='gray')
ax.coastlines()
fig.savefig('map.png')
