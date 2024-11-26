def select_loader(loaders, id):
    selected_loader=None
    for loader in loaders:
        if loader.ids[0] == id:
            selected_loader=loader
            break

    return selected_loader