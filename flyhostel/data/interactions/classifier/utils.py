def select_loader(loaders, identity):
    selected_loader=None
    for loader in loaders:
        if loader.identity == identity:
            selected_loader=loader
            break

    if selected_loader is None:
        import ipdb; ipdb.set_trace()
    return selected_loader