# Map Plotting script
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point

def plot_f_map(
    f: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    central_longitude: float = 180,
    *,
    levels=None,
    cmap: str = 'turbo',
    use_pcolormesh: bool = False,
    ocean_bg_color: str = '#eef7ff',
    land_color: str = 'khaki',
    coastline_resolution: str = '110m',
    title: str | None = None,
    colorbar_label: str = 'Depth [m]',
    figsize: tuple = (12, 6),
    zorder_contours: int = 1,
    land_overlay: bool = True,
    draw_gridlines: bool = True
):
    """
    Plot a global map of a 2D field f using Cartopy, centered at a chosen longitude.

    Parameters
    ----------
    f : np.ndarray
        2D array of the field to plot, shaped (nlat, nlon). Positive values expected.
    lon : np.ndarray
        1D array of longitudes in degrees. Can be [0, 360) or [-180, 180].
    lat : np.ndarray
        1D array of latitudes in degrees, typically [-90, 90].
    central_longitude : float, optional
        Center of the display projection (e.g., 0 for Atlantic-centered,
        180 for Pacific-centered). Default is 180.
    levels : sequence or None, optional
        Contour levels. If None, computed from data with 21 levels.
    cmap : str, optional
        Matplotlib colormap name. Default 'turbo'.
    use_pcolormesh : bool, optional
        If True, uses pcolormesh (faster). If False, uses contourf (default).
    ocean_bg_color : str, optional
        Background fill color for oceans.
    land_color : str, optional
        Fill color for land overlay to hide ocean contours over land.
    coastline_resolution : str, optional
        Resolution for coastlines/features ('110m', '50m', '10m').
    title : str or None, optional
        Plot title. If None, no title is set.
    colorbar_label : str, optional
        Label for the colorbar.
    figsize : tuple, optional
        Figure size in inches.
    zorder_contours : int, optional
        Z-order for the filled contours/pcolormesh.
    land_overlay : bool, optional
        If True, overlays land polygons to visually mask ocean over land.
    draw_gridlines : bool, optional
        If True, draws dashed gridlines (labels off for Robinson).

    Returns
    -------
    (fig, ax) : tuple
        Matplotlib figure and axes objects.
    """
    # --- Input validation & orientation ---
    if f.ndim != 2:
        raise ValueError(f"`f` must be 2D (lat, lon). Got shape {f.shape}.")
    if lon.ndim != 1 or lat.ndim != 1:
        raise ValueError("`lon` and `lat` must be 1D arrays.")
    if f.shape != (lat.size, lon.size):
        raise ValueError(
            f"`f` shape {f.shape} must be (lat.size, lon.size)=({lat.size}, {lon.size})."
        )

    # Ensure latitude is ascending (south to north) to match plotting meshgrid convention
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        f = f[::-1, :]

    # --- Add cyclic column along longitude to avoid seam ---
    # axis=1 because f is shaped (lat, lon)
    f_cyc, lon_cyc = add_cyclic_point(f, coord=lon, axis=1)

    # Build 2D meshes for plotting
    lon2d_cyc, lat2d_cyc = np.meshgrid(lon_cyc, lat)

    # --- Map setup ---
    proj_map = ccrs.Robinson(central_longitude=central_longitude)
    proj_data = ccrs.PlateCarree()

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=proj_map)
    ax.set_global()

    # Subtle ocean background
    ax.add_feature(
        cfeature.OCEAN.with_scale(coastline_resolution),
        facecolor=ocean_bg_color,
        edgecolor='none',
        zorder=0
    )

    # --- Levels ---
    if levels is None:
        vmin = float(np.nanmin(f_cyc))
        vmax = float(np.nanmax(f_cyc))
        # Avoid degenerate levels if data are constant
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        levels = np.linspace(vmin, vmax, 21)

    # --- Plotting: contourf or pcolormesh ---
    if use_pcolormesh:
        # shading='auto' avoids dimension mismatch warnings for pcolormesh
        mappable = ax.pcolormesh(
            lon2d_cyc, lat2d_cyc, f_cyc,
            cmap=cmap, shading='auto',
            transform=proj_data, zorder=zorder_contours
        )
    else:
        mappable = ax.contourf(
            lon2d_cyc, lat2d_cyc, f_cyc,
            levels=levels, cmap=cmap, extend='both',
            transform=proj_data, antialiased=True,
            zorder=zorder_contours
        )

    # --- Land overlay & coastlines ---
    if land_overlay:
        ax.add_feature(
            cfeature.LAND.with_scale(coastline_resolution),
            facecolor=land_color, edgecolor='none', zorder=3
        )
    ax.coastlines(resolution=coastline_resolution, linewidth=0.8, zorder=4)

    # --- Gridlines ---
    if draw_gridlines:
        ax.gridlines(draw_labels=False, linewidth=0.5,
                     color='gray', alpha=0.5, linestyle='--')

    # --- Colorbar & title ---
    cbar = plt.colorbar(mappable, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7)
    if colorbar_label:
        cbar.set_label(colorbar_label)
    if title:
        ax.set_title(title)

    plt.tight_layout()
    return fig, ax
