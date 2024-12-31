from App import dbc,html



def render_table(result:dict):
    knn_std=result.get('KNN').get('with_std')
    knn_without_std=result.get('KNN').get('without_std')
    mlp_std=result.get('MLP').get('with_std')
    mlp_without_std=result.get('MLP').get('without_std')
    container =dbc.Container([
        dbc.Table([
            html.Thead([
                html.Tr([
                    html.Th("",rowSpan=2,className="text-center"),
                    html.Th("K-Neirest-Neihbord",colSpan=2,className="text-center"),
                    html.Th("MLP-Classifier",colSpan=2,className="text-center")
                ]),
                html.Tr([
                    html.Th("with standardisation",className="text-center"),
                    html.Th("without standardisation",className="text-center"),
                    html.Th("with standardisation",className="text-center"),
                    html.Th("without standardisation",className="text-center"),
                ]),
            ]),
            html.Tbody([
                html.Tr([
                    html.Td("accuracy",className="fs-5 fw-bold text-center"),
                    html.Td(f"{knn_std.get('accuracy').get('Fitness')}%",className="text-center"),
                    html.Td(f"{knn_without_std.get('accuracy').get('Fitness')}%",className="text-center"),
                    html.Td(f"{mlp_std.get('accuracy').get('Fitness')}%",className="text-center"),
                    html.Td(f"{mlp_without_std.get('accuracy').get('Fitness')}%",className="text-center"),
                ]),
                html.Tr([
                    html.Td("f1_score",className="fs-5 fw-bold text-center"),
                    html.Td(f"{knn_std.get('f1_score').get('Fitness')}%",className="text-center"),
                    html.Td(f"{knn_without_std.get('f1_score').get('Fitness')}%",className="text-center"),
                    html.Td(f"{mlp_std.get('f1_score').get('Fitness')}%",className="text-center"),
                    html.Td(f"{mlp_without_std.get('f1_score').get('Fitness')}%",className="text-center"),
                ]),
                html.Tr([
                    html.Td("recall",className="fs-5 fw-bold text-center"),
                    html.Td(f"{knn_std.get('recall').get('Fitness')}%",className="text-center"),
                    html.Td(f"{knn_without_std.get('recall').get('Fitness')}%",className="text-center"),
                    html.Td(f"{mlp_std.get('recall').get('Fitness')}%",className="text-center"),
                    html.Td(f"{mlp_without_std.get('recall').get('Fitness')}%",className="text-center"),
                ]),
                html.Tr([
                    html.Td("precision",className="fs-5 fw-bold text-center"),
                    html.Td(f"{knn_std.get('precision').get('Fitness')}%",className="text-center"),
                    html.Td(f"{knn_without_std.get('precision').get('Fitness')}%",className="text-center"),
                    html.Td(f"{mlp_std.get('precision').get('Fitness')}%",className="text-center"),
                    html.Td(f"{mlp_without_std.get('precision').get('Fitness')}%",className="text-center"),
                ]),
            ])
            
        ],responsive=True,hover=True )
    ],className="px-5 py-3 my-5")
    return container