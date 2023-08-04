import plotly.graph_objs as go


class MeshViewer:
    def __init__(self):
        self.data = []

    def add_mesh(self, V, F=None, **kwargs):
        mesh = go.Mesh3d(
            x=V[:,0],
            y=V[:,1],
            z=V[:,2],
            # i, j and k give the vertices of triangles
            i = F[:,0] if F is not None else None,
            j = F[:,1] if F is not None else None,
            k = F[:,2] if F is not None else None,
            showscale=True,
        )
        self.data.append(mesh)
    
    def show(self, clear=True, **kwargs):
        fig = go.Figure(data=self.data)
        fig.show()
        if clear:
            self.data = []
