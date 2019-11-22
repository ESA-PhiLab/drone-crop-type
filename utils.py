from random import choice
import os

# random names to name the different experiments:
names = [
    'Aaren', 'Adah', 'Adelice', 'Adrea', 'Agace', 'Aida', 'Aimee', 'Albertina',
    'Alexa', 'Alice', 'Alisun', 'Allsun', 'Aloysia', 'Alyss', 'Amara', 'Amil',
    'Anastasia', 'Andriana', 'Angelika', 'Anjela', 'Annalee', 'Annice',
    'Anthiathia', 'Arabele', 'Ardisj', 'Arlee', 'Arlyn', 'Ashly', 'Aubrie',
    'Aundrea', 'Austina', 'Babb', 'Barbee', 'Beatrisa', 'Belicia', 'Benita',
    'Bernelle', 'Bertha', 'Bethena', 'Bev', 'Bidget', 'Blaire', 'Blondelle',
    'Bonita', 'Breanne', 'Bridget', 'Brit', 'Brook', 'Cacilia', 'Cam',
    'Candice', 'Carey', 'Carleen', 'Carlynne', 'Caro', 'Caron', 'Cassandra',
    'Cathe', 'Catie', 'Cecilla', 'Celinda', 'Charin', 'Charmain', 'Chere',
    'Cherry', 'Chrissie', 'Christie', 'Cilka', 'Clarabelle', 'Clarita',
    'Clementine', 'Cody', 'Concordia', 'Cora', 'Coreen', 'Corissa', 'Corrinne',
    'Cristabel', 'Cybil', 'Dacy', 'Dalila', 'Daniele', 'Daphene', 'Darelle',
    'Daryl', 'Dawna', 'Debi', 'Deerdre', 'Della', 'Denise', 'Devi',
    'Diane-Marie', 'Dione', 'Dollie', 'Donica', 'Dorelia', 'Dorisa', 'Dorry',
    'Druci', 'Dulcinea', 'Eadith', 'Eden', 'Eileen', 'Eleanore', 'Elianore',
    'Elizabeth', 'Elmira', 'Elsinore', 'Ema', 'Emily', 'Emmye', 'Erica',
    'Ernaline', 'Estell', 'Etti', 'Evangelin', 'Evvy', 'Fania', 'Faustine',
    'Felicdad', 'Fernande', 'Fiona', 'Florenza', 'Flossy', 'Frannie',
    'Fredrika', 'Gabrielle', 'Gay', 'Genna', 'Georgiana', 'Germaine',
    'Giacinta', 'Gillie', 'Gisela', 'Glenine', 'Godiva', 'Grayce', 'Guenevere',
    'Gusti', 'Gwyn', 'Hallie', 'Harmonie', 'Heath', 'Helaina', 'Henrie',
    'Hestia', 'Holli', 'Hyacinthe', 'Ileane', 'Indira', 'Iolande', 'Isabelle',
    'Ivy', 'Jaclin', 'Jaime', 'Janeen', 'Janice', 'Jaquenetta', 'Jeanette',
    'Jen', 'Jenni', 'Jerry', 'Jewel', 'Jo', 'Jobina', 'Joell', 'Joice',
    'Jonis', 'Josephina', 'Jsandye', 'Juliann', 'Junia', 'Kailey', 'Kally',
    'Karel', 'Karla', 'Karolina', 'Kassi', 'Katharyn', 'Katie', 'Kay',
    'Kelcy', 'Kendre', 'Ketti', 'Kimberli', 'Kirby', 'Kizzie', 'Kori',
    'Kristien', 'Kylila', 'Lana', 'Larine', 'Lauree', 'Laverna', 'Lebbie',
    'Leilah', 'Leola', 'Leslie', 'Lexine', 'Libby', 'Lilli', 'Lindsy',
    'Lisbeth', 'Livvie', 'Loleta', 'Loree', 'Lorilee', 'Lotte', 'Lucie',
    'Merrielle', 'Michaella', 'Miguelita', 'Mimi', 'Minta', 'Mirilla',
    'Mollee', 'Tabina', 'Trina', 'Tuesday', 'Ursala', 'Valentina', 'Vanda',
    'Vere', 'Vi', 'Vinnie', 'Vittoria', 'Vivyanne', 'Wenda', 'Willamina',
    'Winifred', 'Wynnie', 'Ynes', 'Zandra', 'Zoe'
]

def get_experiment_name(path):
    """Create an unique experiment name

    Args:
        path: Path where the experiment should be saved. This function takes
        care that there are no other files inside this path which names conflict
        with the new experiment name.

    Returns:
        A string with a new experiment name
    """
    counter = 0
    base_name = None
    while True:
        if base_name is None:
            name = choice(names)
        else:
            name = base_name + "-" + choice(names)
        
        if not os.path.exists(os.path.join(path, name)):
            return name
        else:
            # Make sure we won't run into infinite loops:
            counter += 1
            if counter > len(names):
                base_name = choice(names)
