ORM Quick Start
For new users who want to quickly see what basic ORM use looks like, here’s an abbreviated form of the mappings and examples used in the SQLAlchemy Unified Tutorial. The code here is fully runnable from a clean command line.

As the descriptions in this section are intentionally very short, please proceed to the full SQLAlchemy Unified Tutorial for a much more in-depth description of each of the concepts being illustrated here.

Changed in version 2.0: The ORM Quickstart is updated for the latest PEP 484-aware features using new constructs including mapped_column(). See the section ORM Declarative Models for migration information.

Declare Models
Here, we define module-level constructs that will form the structures which we will be querying from the database. This structure, known as a Declarative Mapping, defines at once both a Python object model, as well as database metadata that describes real SQL tables that exist, or will exist, in a particular database:

>>> from typing import List
>>> from typing import Optional
>>> from sqlalchemy import ForeignKey
>>> from sqlalchemy import String
>>> from sqlalchemy.orm import DeclarativeBase
>>> from sqlalchemy.orm import Mapped
>>> from sqlalchemy.orm import mapped_column
>>> from sqlalchemy.orm import relationship

>>> class Base(DeclarativeBase):
...     pass

>>> class User(Base):
...     __tablename__ = "user_account"
...
...     id: Mapped[int] = mapped_column(primary_key=True)
...     name: Mapped[str] = mapped_column(String(30))
...     fullname: Mapped[Optional[str]]
...
...     addresses: Mapped[List["Address"]] = relationship(
...         back_populates="user", cascade="all, delete-orphan"
...     )
...
...     def __repr__(self) -> str:
...         return f"User(id={self.id!r}, name={self.name!r}, fullname={self.fullname!r})"

>>> class Address(Base):
...     __tablename__ = "address"
...
...     id: Mapped[int] = mapped_column(primary_key=True)
...     email_address: Mapped[str]
...     user_id: Mapped[int] = mapped_column(ForeignKey("user_account.id"))
...
...     user: Mapped["User"] = relationship(back_populates="addresses")
...
...     def __repr__(self) -> str:
...         return f"Address(id={self.id!r}, email_address={self.email_address!r})"
The mapping starts with a base class, which above is called Base, and is created by making a simple subclass against the DeclarativeBase class.

Individual mapped classes are then created by making subclasses of Base. A mapped class typically refers to a single particular database table, the name of which is indicated by using the __tablename__ class-level attribute.

Next, columns that are part of the table are declared, by adding attributes that include a special typing annotation called Mapped. The name of each attribute corresponds to the column that is to be part of the database table. The datatype of each column is taken first from the Python datatype that’s associated with each Mapped annotation; int for INTEGER, str for VARCHAR, etc. Nullability derives from whether or not the Optional[] type modifier is used. More specific typing information may be indicated using SQLAlchemy type objects in the right side mapped_column() directive, such as the String datatype used above in the User.name column. The association between Python types and SQL types can be customized using the type annotation map.

The mapped_column() directive is used for all column-based attributes that require more specific customization. Besides typing information, this directive accepts a wide variety of arguments that indicate specific details about a database column, including server defaults and constraint information, such as membership within the primary key and foreign keys. The mapped_column() directive accepts a superset of arguments that are accepted by the SQLAlchemy Column class, which is used by SQLAlchemy Core to represent database columns.

All ORM mapped classes require at least one column be declared as part of the primary key, typically by using the Column.primary_key parameter on those mapped_column() objects that should be part of the key. In the above example, the User.id and Address.id columns are marked as primary key.

Taken together, the combination of a string table name as well as a list of column declarations is known in SQLAlchemy as table metadata. Setting up table metadata using both Core and ORM approaches is introduced in the SQLAlchemy Unified Tutorial at Working with Database Metadata. The above mapping is an example of what’s known as Annotated Declarative Table configuration.

Other variants of Mapped are available, most commonly the relationship() construct indicated above. In contrast to the column-based attributes, relationship() denotes a linkage between two ORM classes. In the above example, User.addresses links User to Address, and Address.user links Address to User. The relationship() construct is introduced in the SQLAlchemy Unified Tutorial at Working with ORM Related Objects.

Finally, the above example classes include a __repr__() method, which is not required but is useful for debugging. Mapped classes can be created with methods such as __repr__() generated automatically, using dataclasses. More on dataclass mapping at Declarative Dataclass Mapping.

Create an Engine
The Engine is a factory that can create new database connections for us, which also holds onto connections inside of a Connection Pool for fast reuse. For learning purposes, we normally use a SQLite memory-only database for convenience:

>>> from sqlalchemy import create_engine
>>> engine = create_engine("sqlite://", echo=True)
Tip

The echo=True parameter indicates that SQL emitted by connections will be logged to standard out.

A full intro to the Engine starts at Establishing Connectivity - the Engine.

Emit CREATE TABLE DDL
Using our table metadata and our engine, we can generate our schema at once in our target SQLite database, using a method called MetaData.create_all():

>>> Base.metadata.create_all(engine)
BEGIN (implicit)
PRAGMA main.table_...info("user_account")
...
PRAGMA main.table_...info("address")
...
CREATE TABLE user_account (
    id INTEGER NOT NULL,
    name VARCHAR(30) NOT NULL,
    fullname VARCHAR,
    PRIMARY KEY (id)
)
...
CREATE TABLE address (
    id INTEGER NOT NULL,
    email_address VARCHAR NOT NULL,
    user_id INTEGER NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY(user_id) REFERENCES user_account (id)
)
...
COMMIT
A lot just happened from that bit of Python code we wrote. For a complete overview of what’s going on on with Table metadata, proceed in the Tutorial at Working with Database Metadata.

Create Objects and Persist
We are now ready to insert data in the database. We accomplish this by creating instances of User and Address classes, which have an __init__() method already as established automatically by the declarative mapping process. We then pass them to the database using an object called a Session, which makes use of the Engine to interact with the database. The Session.add_all() method is used here to add multiple objects at once, and the Session.commit() method will be used to flush any pending changes to the database and then commit the current database transaction, which is always in progress whenever the Session is used:

>>> from sqlalchemy.orm import Session

>>> with Session(engine) as session:
...     spongebob = User(
...         name="spongebob",
...         fullname="Spongebob Squarepants",
...         addresses=[Address(email_address="spongebob@sqlalchemy.org")],
...     )
...     sandy = User(
...         name="sandy",
...         fullname="Sandy Cheeks",
...         addresses=[
...             Address(email_address="sandy@sqlalchemy.org"),
...             Address(email_address="sandy@squirrelpower.org"),
...         ],
...     )
...     patrick = User(name="patrick", fullname="Patrick Star")
...
...     session.add_all([spongebob, sandy, patrick])
...
...     session.commit()
BEGIN (implicit)
INSERT INTO user_account (name, fullname) VALUES (?, ?) RETURNING id
[...] ('spongebob', 'Spongebob Squarepants')
INSERT INTO user_account (name, fullname) VALUES (?, ?) RETURNING id
[...] ('sandy', 'Sandy Cheeks')
INSERT INTO user_account (name, fullname) VALUES (?, ?) RETURNING id
[...] ('patrick', 'Patrick Star')
INSERT INTO address (email_address, user_id) VALUES (?, ?) RETURNING id
[...] ('spongebob@sqlalchemy.org', 1)
INSERT INTO address (email_address, user_id) VALUES (?, ?) RETURNING id
[...] ('sandy@sqlalchemy.org', 2)
INSERT INTO address (email_address, user_id) VALUES (?, ?) RETURNING id
[...] ('sandy@squirrelpower.org', 2)
COMMIT
Tip

It’s recommended that the Session be used in context manager style as above, that is, using the Python with: statement. The Session object represents active database resources so it’s good to make sure it’s closed out when a series of operations are completed. In the next section, we’ll keep a Session opened just for illustration purposes.

Basics on creating a Session are at Executing with an ORM Session and more at Basics of Using a Session.

Then, some varieties of basic persistence operations are introduced at Inserting Rows using the ORM Unit of Work pattern.

Simple SELECT
With some rows in the database, here’s the simplest form of emitting a SELECT statement to load some objects. To create SELECT statements, we use the select() function to create a new Select object, which we then invoke using a Session. The method that is often useful when querying for ORM objects is the Session.scalars() method, which will return a ScalarResult object that will iterate through the ORM objects we’ve selected:

>>> from sqlalchemy import select

>>> session = Session(engine)

>>> stmt = select(User).where(User.name.in_(["spongebob", "sandy"]))

>>> for user in session.scalars(stmt):
...     print(user)
BEGIN (implicit)
SELECT user_account.id, user_account.name, user_account.fullname
FROM user_account
WHERE user_account.name IN (?, ?)
[...] ('spongebob', 'sandy')
User(id=1, name='spongebob', fullname='Spongebob Squarepants')
User(id=2, name='sandy', fullname='Sandy Cheeks')
The above query also made use of the Select.where() method to add WHERE criteria, and also used the ColumnOperators.in_() method that’s part of all SQLAlchemy column-like constructs to use the SQL IN operator.

More detail on how to select objects and individual columns is at Selecting ORM Entities and Columns.

SELECT with JOIN
It’s very common to query amongst multiple tables at once, and in SQL the JOIN keyword is the primary way this happens. The Select construct creates joins using the Select.join() method:

>>> stmt = (
...     select(Address)
...     .join(Address.user)
...     .where(User.name == "sandy")
...     .where(Address.email_address == "sandy@sqlalchemy.org")
... )
>>> sandy_address = session.scalars(stmt).one()
SELECT address.id, address.email_address, address.user_id
FROM address JOIN user_account ON user_account.id = address.user_id
WHERE user_account.name = ? AND address.email_address = ?
[...] ('sandy', 'sandy@sqlalchemy.org')
>>> sandy_address
Address(id=2, email_address='sandy@sqlalchemy.org')
The above query illustrates multiple WHERE criteria which are automatically chained together using AND, as well as how to use SQLAlchemy column-like objects to create “equality” comparisons, which uses the overridden Python method ColumnOperators.__eq__() to produce a SQL criteria object.

Some more background on the concepts above are at The WHERE clause and Explicit FROM clauses and JOINs.

Make Changes
The Session object, in conjunction with our ORM-mapped classes User and Address, automatically track changes to the objects as they are made, which result in SQL statements that will be emitted the next time the Session flushes. Below, we change one email address associated with “sandy”, and also add a new email address to “patrick”, after emitting a SELECT to retrieve the row for “patrick”:

>>> stmt = select(User).where(User.name == "patrick")
>>> patrick = session.scalars(stmt).one()
SELECT user_account.id, user_account.name, user_account.fullname
FROM user_account
WHERE user_account.name = ?
[...] ('patrick',)
>>> patrick.addresses.append(Address(email_address="patrickstar@sqlalchemy.org"))
SELECT address.id AS address_id, address.email_address AS address_email_address, address.user_id AS address_user_id
FROM address
WHERE ? = address.user_id
[...] (3,)
>>> sandy_address.email_address = "sandy_cheeks@sqlalchemy.org"

>>> session.commit()
UPDATE address SET email_address=? WHERE address.id = ?
[...] ('sandy_cheeks@sqlalchemy.org', 2)
INSERT INTO address (email_address, user_id) VALUES (?, ?)
[...] ('patrickstar@sqlalchemy.org', 3)
COMMIT
Notice when we accessed patrick.addresses, a SELECT was emitted. This is called a lazy load. Background on different ways to access related items using more or less SQL is introduced at Loader Strategies.

A detailed walkthrough on ORM data manipulation starts at Data Manipulation with the ORM.

Some Deletes
All things must come to an end, as is the case for some of our database rows - here’s a quick demonstration of two different forms of deletion, both of which are important based on the specific use case.

First we will remove one of the Address objects from the “sandy” user. When the Session next flushes, this will result in the row being deleted. This behavior is something that we configured in our mapping called the delete cascade. We can get a handle to the sandy object by primary key using Session.get(), then work with the object:

>>> sandy = session.get(User, 2)
BEGIN (implicit)
SELECT user_account.id AS user_account_id, user_account.name AS user_account_name, user_account.fullname AS user_account_fullname
FROM user_account
WHERE user_account.id = ?
[...] (2,)
>>> sandy.addresses.remove(sandy_address)
SELECT address.id AS address_id, address.email_address AS address_email_address, address.user_id AS address_user_id
FROM address
WHERE ? = address.user_id
[...] (2,)
The last SELECT above was the lazy load operation proceeding so that the sandy.addresses collection could be loaded, so that we could remove the sandy_address member. There are other ways to go about this series of operations that won’t emit as much SQL.

We can choose to emit the DELETE SQL for what’s set to be changed so far, without committing the transaction, using the Session.flush() method:

>>> session.flush()
DELETE FROM address WHERE address.id = ?
[...] (2,)
Next, we will delete the “patrick” user entirely. For a top-level delete of an object by itself, we use the Session.delete() method; this method doesn’t actually perform the deletion, but sets up the object to be deleted on the next flush. The operation will also cascade to related objects based on the cascade options that we configured, in this case, onto the related Address objects:

>>> session.delete(patrick)
SELECT user_account.id AS user_account_id, user_account.name AS user_account_name, user_account.fullname AS user_account_fullname
FROM user_account
WHERE user_account.id = ?
[...] (3,)
SELECT address.id AS address_id, address.email_address AS address_email_address, address.user_id AS address_user_id
FROM address
WHERE ? = address.user_id
[...] (3,)
The Session.delete() method in this particular case emitted two SELECT statements, even though it didn’t emit a DELETE, which might seem surprising. This is because when the method went to inspect the object, it turns out the patrick object was expired, which happened when we last called upon Session.commit(), and the SQL emitted was to re-load the rows from the new transaction. This expiration is optional, and in normal use we will often be turning it off for situations where it doesn’t apply well.

To illustrate the rows being deleted, here’s the commit:

>>> session.commit()
DELETE FROM address WHERE address.id = ?
[...] (4,)
DELETE FROM user_account WHERE user_account.id = ?
[...] (3,)
COMMIT
The Tutorial discusses ORM deletion at Deleting ORM Objects using the Unit of Work pattern. Background on object expiration is at Expiring / Refreshing; cascades are discussed in depth at Cascades.

Learn the above concepts in depth
For a new user, the above sections were likely a whirlwind tour. There’s a lot of important concepts in each step above that weren’t covered. With a quick overview of what things look like, it’s recommended to work through the SQLAlchemy Unified Tutorial to gain a solid working knowledge of what’s really going on above. Good luck!

Previous: SQLAlchemy ORM Next: ORM Mapped Class Configuration
© Copyright 2007-2025, the SQLAlchemy authors and contributors.
flambé! the dragon and The Alchemist image designs created and generously donated by Rotem Yaari.

Created using Sphinx 7.2.6. Documentation last generated: Fri 07 Mar 2025 09:40:25 AM EST


ORM Mapped Class Overview
Overview of ORM class mapping configuration.

For readers new to the SQLAlchemy ORM and/or new to Python in general, it’s recommended to browse through the ORM Quick Start and preferably to work through the SQLAlchemy Unified Tutorial, where ORM configuration is first introduced at Using ORM Declarative Forms to Define Table Metadata.

ORM Mapping Styles
SQLAlchemy features two distinct styles of mapper configuration, which then feature further sub-options for how they are set up. The variability in mapper styles is present to suit a varied list of developer preferences, including the degree of abstraction of a user-defined class from how it is to be mapped to relational schema tables and columns, what kinds of class hierarchies are in use, including whether or not custom metaclass schemes are present, and finally if there are other class-instrumentation approaches present such as if Python dataclasses are in use simultaneously.

In modern SQLAlchemy, the difference between these styles is mostly superficial; when a particular SQLAlchemy configurational style is used to express the intent to map a class, the internal process of mapping the class proceeds in mostly the same way for each, where the end result is always a user-defined class that has a Mapper configured against a selectable unit, typically represented by a Table object, and the class itself has been instrumented to include behaviors linked to relational operations both at the level of the class as well as on instances of that class. As the process is basically the same in all cases, classes mapped from different styles are always fully interoperable with each other. The protocol MappedClassProtocol can be used to indicate a mapped class when using type checkers such as mypy.

The original mapping API is commonly referred to as “classical” style, whereas the more automated style of mapping is known as “declarative” style. SQLAlchemy now refers to these two mapping styles as imperative mapping and declarative mapping.

Regardless of what style of mapping used, all ORM mappings as of SQLAlchemy 1.4 originate from a single object known as registry, which is a registry of mapped classes. Using this registry, a set of mapper configurations can be finalized as a group, and classes within a particular registry may refer to each other by name within the configurational process.

Changed in version 1.4: Declarative and classical mapping are now referred to as “declarative” and “imperative” mapping, and are unified internally, all originating from the registry construct that represents a collection of related mappings.

Declarative Mapping
The Declarative Mapping is the typical way that mappings are constructed in modern SQLAlchemy. The most common pattern is to first construct a base class using the DeclarativeBase superclass. The resulting base class, when subclassed will apply the declarative mapping process to all subclasses that derive from it, relative to a particular registry that is local to the new base by default. The example below illustrates the use of a declarative base which is then used in a declarative table mapping:

from sqlalchemy import Integer, String, ForeignKey
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column


# declarative base class
class Base(DeclarativeBase):
    pass


# an example mapping using the base
class User(Base):
    __tablename__ = "user"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    fullname: Mapped[str] = mapped_column(String(30))
    nickname: Mapped[Optional[str]]
Above, the DeclarativeBase class is used to generate a new base class (within SQLAlchemy’s documentation it’s typically referred to as Base, however can have any desired name) from which new classes to be mapped may inherit from, as above a new mapped class User is constructed.

Changed in version 2.0: The DeclarativeBase superclass supersedes the use of the declarative_base() function and registry.generate_base() methods; the superclass approach integrates with PEP 484 tools without the use of plugins. See ORM Declarative Models for migration notes.

The base class refers to a registry object that maintains a collection of related mapped classes. as well as to a MetaData object that retains a collection of Table objects to which the classes are mapped.

The major Declarative mapping styles are further detailed in the following sections:

Using a Declarative Base Class - declarative mapping using a base class.

Declarative Mapping using a Decorator (no declarative base) - declarative mapping using a decorator, rather than a base class.

Within the scope of a Declarative mapped class, there are also two varieties of how the Table metadata may be declared. These include:

Declarative Table with mapped_column() - table columns are declared inline within the mapped class using the mapped_column() directive (or in legacy form, using the Column object directly). The mapped_column() directive may also be optionally combined with type annotations using the Mapped class which can provide some details about the mapped columns directly. The column directives, in combination with the __tablename__ and optional __table_args__ class level directives will allow the Declarative mapping process to construct a Table object to be mapped.

Declarative with Imperative Table (a.k.a. Hybrid Declarative) - Instead of specifying table name and attributes separately, an explicitly constructed Table object is associated with a class that is otherwise mapped declaratively. This style of mapping is a hybrid of “declarative” and “imperative” mapping, and applies to techniques such as mapping classes to reflected Table objects, as well as mapping classes to existing Core constructs such as joins and subqueries.

Documentation for Declarative mapping continues at Mapping Classes with Declarative.

Imperative Mapping
An imperative or classical mapping refers to the configuration of a mapped class using the registry.map_imperatively() method, where the target class does not include any declarative class attributes.

Tip

The imperative mapping form is a lesser-used form of mapping that originates from the very first releases of SQLAlchemy in 2006. It’s essentially a means of bypassing the Declarative system to provide a more “barebones” system of mapping, and does not offer modern features such as PEP 484 support. As such, most documentation examples use Declarative forms, and it’s recommended that new users start with Declarative Table configuration.

Changed in version 2.0: The registry.map_imperatively() method is now used to create classical mappings. The sqlalchemy.orm.mapper() standalone function is effectively removed.

In “classical” form, the table metadata is created separately with the Table construct, then associated with the User class via the registry.map_imperatively() method, after establishing a registry instance. Normally, a single instance of registry shared for all mapped classes that are related to each other:

from sqlalchemy import Table, Column, Integer, String, ForeignKey
from sqlalchemy.orm import registry

mapper_registry = registry()

user_table = Table(
    "user",
    mapper_registry.metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(50)),
    Column("fullname", String(50)),
    Column("nickname", String(12)),
)


class User:
    pass


mapper_registry.map_imperatively(User, user_table)
Information about mapped attributes, such as relationships to other classes, are provided via the properties dictionary. The example below illustrates a second Table object, mapped to a class called Address, then linked to User via relationship():

address = Table(
    "address",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("user_id", Integer, ForeignKey("user.id")),
    Column("email_address", String(50)),
)

mapper_registry.map_imperatively(
    User,
    user,
    properties={
        "addresses": relationship(Address, backref="user", order_by=address.c.id)
    },
)

mapper_registry.map_imperatively(Address, address)
Note that classes which are mapped with the Imperative approach are fully interchangeable with those mapped with the Declarative approach. Both systems ultimately create the same configuration, consisting of a Table, user-defined class, linked together with a Mapper object. When we talk about “the behavior of Mapper”, this includes when using the Declarative system as well - it’s still used, just behind the scenes.

Mapped Class Essential Components
With all mapping forms, the mapping of the class can be configured in many ways by passing construction arguments that ultimately become part of the Mapper object via its constructor. The parameters that are delivered to Mapper originate from the given mapping form, including parameters passed to registry.map_imperatively() for an Imperative mapping, or when using the Declarative system, from a combination of the table columns, SQL expressions and relationships being mapped along with that of attributes such as __mapper_args__.

There are four general classes of configuration information that the Mapper class looks for:

The class to be mapped
This is a class that we construct in our application. There are generally no restrictions on the structure of this class. [1] When a Python class is mapped, there can only be one Mapper object for the class. [2]

When mapping with the declarative mapping style, the class to be mapped is either a subclass of the declarative base class, or is handled by a decorator or function such as registry.mapped().

When mapping with the imperative style, the class is passed directly as the map_imperatively.class_ argument.

The table, or other from clause object
In the vast majority of common cases this is an instance of Table. For more advanced use cases, it may also refer to any kind of FromClause object, the most common alternative objects being the Subquery and Join object.

When mapping with the declarative mapping style, the subject table is either generated by the declarative system based on the __tablename__ attribute and the Column objects presented, or it is established via the __table__ attribute. These two styles of configuration are presented at Declarative Table with mapped_column() and Declarative with Imperative Table (a.k.a. Hybrid Declarative).

When mapping with the imperative style, the subject table is passed positionally as the map_imperatively.local_table argument.

In contrast to the “one mapper per class” requirement of a mapped class, the Table or other FromClause object that is the subject of the mapping may be associated with any number of mappings. The Mapper applies modifications directly to the user-defined class, but does not modify the given Table or other FromClause in any way.

The properties dictionary
This is a dictionary of all of the attributes that will be associated with the mapped class. By default, the Mapper generates entries for this dictionary derived from the given Table, in the form of ColumnProperty objects which each refer to an individual Column of the mapped table. The properties dictionary will also contain all the other kinds of MapperProperty objects to be configured, most commonly instances generated by the relationship() construct.

When mapping with the declarative mapping style, the properties dictionary is generated by the declarative system by scanning the class to be mapped for appropriate attributes. See the section Defining Mapped Properties with Declarative for notes on this process.

When mapping with the imperative style, the properties dictionary is passed directly as the properties parameter to registry.map_imperatively(), which will pass it along to the Mapper.properties parameter.

Other mapper configuration parameters
When mapping with the declarative mapping style, additional mapper configuration arguments are configured via the __mapper_args__ class attribute. Examples of use are available at Mapper Configuration Options with Declarative.

When mapping with the imperative style, keyword arguments are passed to the to registry.map_imperatively() method which passes them along to the Mapper class.

The full range of parameters accepted are documented at Mapper.

Mapped Class Behavior
Across all styles of mapping using the registry object, the following behaviors are common:

Default Constructor
The registry applies a default constructor, i.e. __init__ method, to all mapped classes that don’t explicitly have their own __init__ method. The behavior of this method is such that it provides a convenient keyword constructor that will accept as optional keyword arguments all the attributes that are named. E.g.:

from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "user"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    fullname: Mapped[str]
An object of type User above will have a constructor which allows User objects to be created as:

u1 = User(name="some name", fullname="some fullname")
Tip

The Declarative Dataclass Mapping feature provides an alternate means of generating a default __init__() method by using Python dataclasses, and allows for a highly configurable constructor form.

Warning

The __init__() method of the class is called only when the object is constructed in Python code, and not when an object is loaded or refreshed from the database. See the next section Maintaining Non-Mapped State Across Loads for a primer on how to invoke special logic when objects are loaded.

A class that includes an explicit __init__() method will maintain that method, and no default constructor will be applied.

To change the default constructor used, a user-defined Python callable may be provided to the registry.constructor parameter which will be used as the default constructor.

The constructor also applies to imperative mappings:

from sqlalchemy.orm import registry

mapper_registry = registry()

user_table = Table(
    "user",
    mapper_registry.metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(50)),
)


class User:
    pass


mapper_registry.map_imperatively(User, user_table)
The above class, mapped imperatively as described at Imperative Mapping, will also feature the default constructor associated with the registry.

New in version 1.4: classical mappings now support a standard configuration-level constructor when they are mapped via the registry.map_imperatively() method.

Maintaining Non-Mapped State Across Loads
The __init__() method of the mapped class is invoked when the object is constructed directly in Python code:

u1 = User(name="some name", fullname="some fullname")
However, when an object is loaded using the ORM Session, the __init__() method is not called:

u1 = session.scalars(select(User).where(User.name == "some name")).first()
The reason for this is that when loaded from the database, the operation used to construct the object, in the above example the User, is more analogous to deserialization, such as unpickling, rather than initial construction. The majority of the object’s important state is not being assembled for the first time, it’s being re-loaded from database rows.

Therefore to maintain state within the object that is not part of the data that’s stored to the database, such that this state is present when objects are loaded as well as constructed, there are two general approaches detailed below.

Use Python descriptors like @property, rather than state, to dynamically compute attributes as needed.

For simple attributes, this is the simplest approach and the least error prone. For example if an object Point with Point.x and Point.y wanted an attribute with the sum of these attributes:

class Point(Base):
    __tablename__ = "point"
    id: Mapped[int] = mapped_column(primary_key=True)
    x: Mapped[int]
    y: Mapped[int]

    @property
    def x_plus_y(self):
        return self.x + self.y
An advantage of using dynamic descriptors is that the value is computed every time, meaning it maintains the correct value as the underlying attributes (x and y in this case) might change.

Other forms of the above pattern include Python standard library cached_property decorator (which is cached, and not re-computed each time), as well as SQLAlchemy’s hybrid_property decorator which allows for attributes that can work for SQL querying as well.

Establish state on-load using InstanceEvents.load(), and optionally supplemental methods InstanceEvents.refresh() and InstanceEvents.refresh_flush().

These are event hooks that are invoked whenever the object is loaded from the database, or when it is refreshed after being expired. Typically only the InstanceEvents.load() is needed, since non-mapped local object state is not affected by expiration operations. To revise the Point example above looks like:

from sqlalchemy import event


class Point(Base):
    __tablename__ = "point"
    id: Mapped[int] = mapped_column(primary_key=True)
    x: Mapped[int]
    y: Mapped[int]

    def __init__(self, x, y, **kw):
        super().__init__(x=x, y=y, **kw)
        self.x_plus_y = x + y


@event.listens_for(Point, "load")
def receive_load(target, context):
    target.x_plus_y = target.x + target.y
If using the refresh events as well, the event hooks can be stacked on top of one callable if needed, as:

@event.listens_for(Point, "load")
@event.listens_for(Point, "refresh")
@event.listens_for(Point, "refresh_flush")
def receive_load(target, context, attrs=None):
    target.x_plus_y = target.x + target.y
Above, the attrs attribute will be present for the refresh and refresh_flush events and indicate a list of attribute names that are being refreshed.

Runtime Introspection of Mapped classes, Instances and Mappers
A class that is mapped using registry will also feature a few attributes that are common to all mappings:

The __mapper__ attribute will refer to the Mapper that is associated with the class:

mapper = User.__mapper__
This Mapper is also what’s returned when using the inspect() function against the mapped class:

from sqlalchemy import inspect

mapper = inspect(User)
The __table__ attribute will refer to the Table, or more generically to the FromClause object, to which the class is mapped:

table = User.__table__
This FromClause is also what’s returned when using the Mapper.local_table attribute of the Mapper:

table = inspect(User).local_table
For a single-table inheritance mapping, where the class is a subclass that does not have a table of its own, the Mapper.local_table attribute as well as the .__table__ attribute will be None. To retrieve the “selectable” that is actually selected from during a query for this class, this is available via the Mapper.selectable attribute:

table = inspect(User).selectable
Inspection of Mapper objects
As illustrated in the previous section, the Mapper object is available from any mapped class, regardless of method, using the Runtime Inspection API system. Using the inspect() function, one can acquire the Mapper from a mapped class:

>>> from sqlalchemy import inspect
>>> insp = inspect(User)
Detailed information is available including Mapper.columns:

>>> insp.columns
<sqlalchemy.util._collections.OrderedProperties object at 0x102f407f8>
This is a namespace that can be viewed in a list format or via individual names:

>>> list(insp.columns)
[Column('id', Integer(), table=<user>, primary_key=True, nullable=False), Column('name', String(length=50), table=<user>), Column('fullname', String(length=50), table=<user>), Column('nickname', String(length=50), table=<user>)]
>>> insp.columns.name
Column('name', String(length=50), table=<user>)
Other namespaces include Mapper.all_orm_descriptors, which includes all mapped attributes as well as hybrids, association proxies:

>>> insp.all_orm_descriptors
<sqlalchemy.util._collections.ImmutableProperties object at 0x1040e2c68>
>>> insp.all_orm_descriptors.keys()
['fullname', 'nickname', 'name', 'id']
As well as Mapper.column_attrs:

>>> list(insp.column_attrs)
[<ColumnProperty at 0x10403fde0; id>, <ColumnProperty at 0x10403fce8; name>, <ColumnProperty at 0x1040e9050; fullname>, <ColumnProperty at 0x1040e9148; nickname>]
>>> insp.column_attrs.name
<ColumnProperty at 0x10403fce8; name>
>>> insp.column_attrs.name.expression
Column('name', String(length=50), table=<user>)
See also

Mapper

Inspection of Mapped Instances
The inspect() function also provides information about instances of a mapped class. When applied to an instance of a mapped class, rather than the class itself, the object returned is known as InstanceState, which will provide links to not only the Mapper in use by the class, but also a detailed interface that provides information on the state of individual attributes within the instance including their current value and how this relates to what their database-loaded value is.

Given an instance of the User class loaded from the database:

>>> u1 = session.scalars(select(User)).first()
The inspect() function will return to us an InstanceState object:

>>> insp = inspect(u1)
>>> insp
<sqlalchemy.orm.state.InstanceState object at 0x7f07e5fec2e0>
With this object we can see elements such as the Mapper:

>>> insp.mapper
<Mapper at 0x7f07e614ef50; User>
The Session to which the object is attached, if any:

>>> insp.session
<sqlalchemy.orm.session.Session object at 0x7f07e614f160>
Information about the current persistence state for the object:

>>> insp.persistent
True
>>> insp.pending
False
Attribute state information such as attributes that have not been loaded or lazy loaded (assume addresses refers to a relationship() on the mapped class to a related class):

>>> insp.unloaded
{'addresses'}
Information regarding the current in-Python status of attributes, such as attributes that have not been modified since the last flush:

>>> insp.unmodified
{'nickname', 'name', 'fullname', 'id'}
as well as specific history on modifications to attributes since the last flush:

>>> insp.attrs.nickname.value
'nickname'
>>> u1.nickname = "new nickname"
>>> insp.attrs.nickname.history
History(added=['new nickname'], unchanged=(), deleted=['nickname'])
See also

InstanceState

InstanceState.attrs

AttributeState

[1]
When running under Python 2, a Python 2 “old style” class is the only kind of class that isn’t compatible. When running code on Python 2, all classes must extend from the Python object class. Under Python 3 this is always the case.

[2]
There is a legacy feature known as a “non primary mapper”, where additional Mapper objects may be associated with a class that’s already mapped, however they don’t apply instrumentation to the class. This feature is deprecated as of SQLAlchemy 1.3.

Previous: ORM Mapped Class Configuration Next: Mapping Classes with Declarative
© Copyright 2007-2025, the SQLAlchemy authors and contributors.
flambé! the dragon and The Alchemist image designs created and generously donated by Rotem Yaari.

Created using Sphinx 7.2.6. Documentation last generated: Fri 07 Mar 2025 09:40:25 AM EST



Working with ORM Related Objects
In this section, we will cover one more essential ORM concept, which is how the ORM interacts with mapped classes that refer to other objects. In the section Declaring Mapped Classes, the mapped class examples made use of a construct called relationship(). This construct defines a linkage between two different mapped classes, or from a mapped class to itself, the latter of which is called a self-referential relationship.

To describe the basic idea of relationship(), first we’ll review the mapping in short form, omitting the mapped_column() mappings and other directives:

from sqlalchemy.orm import Mapped
from sqlalchemy.orm import relationship


class User(Base):
    __tablename__ = "user_account"

    # ... mapped_column() mappings

    addresses: Mapped[List["Address"]] = relationship(back_populates="user")


class Address(Base):
    __tablename__ = "address"

    # ... mapped_column() mappings

    user: Mapped["User"] = relationship(back_populates="addresses")
Above, the User class now has an attribute User.addresses and the Address class has an attribute Address.user. The relationship() construct, in conjunction with the Mapped construct to indicate typing behavior, will be used to inspect the table relationships between the Table objects that are mapped to the User and Address classes. As the Table object representing the address table has a ForeignKeyConstraint which refers to the user_account table, the relationship() can determine unambiguously that there is a one to many relationship from the User class to the Address class, along the User.addresses relationship; one particular row in the user_account table may be referenced by many rows in the address table.

All one-to-many relationships naturally correspond to a many to one relationship in the other direction, in this case the one noted by Address.user. The relationship.back_populates parameter, seen above configured on both relationship() objects referring to the other name, establishes that each of these two relationship() constructs should be considered to be complimentary to each other; we will see how this plays out in the next section.

Persisting and Loading Relationships
We can start by illustrating what relationship() does to instances of objects. If we make a new User object, we can note that there is a Python list when we access the .addresses element:

>>> u1 = User(name="pkrabs", fullname="Pearl Krabs")
>>> u1.addresses
[]
This object is a SQLAlchemy-specific version of Python list which has the ability to track and respond to changes made to it. The collection also appeared automatically when we accessed the attribute, even though we never assigned it to the object. This is similar to the behavior noted at Inserting Rows using the ORM Unit of Work pattern where it was observed that column-based attributes to which we don’t explicitly assign a value also display as None automatically, rather than raising an AttributeError as would be Python’s usual behavior.

As the u1 object is still transient and the list that we got from u1.addresses has not been mutated (i.e. appended or extended), it’s not actually associated with the object yet, but as we make changes to it, it will become part of the state of the User object.

The collection is specific to the Address class which is the only type of Python object that may be persisted within it. Using the list.append() method we may add an Address object:

>>> a1 = Address(email_address="pearl.krabs@gmail.com")
>>> u1.addresses.append(a1)
At this point, the u1.addresses collection as expected contains the new Address object:

>>> u1.addresses
[Address(id=None, email_address='pearl.krabs@gmail.com')]
As we associated the Address object with the User.addresses collection of the u1 instance, another behavior also occurred, which is that the User.addresses relationship synchronized itself with the Address.user relationship, such that we can navigate not only from the User object to the Address object, we can also navigate from the Address object back to the “parent” User object:

>>> a1.user
User(id=None, name='pkrabs', fullname='Pearl Krabs')
This synchronization occurred as a result of our use of the relationship.back_populates parameter between the two relationship() objects. This parameter names another relationship() for which complementary attribute assignment / list mutation should occur. It will work equally well in the other direction, which is that if we create another Address object and assign to its Address.user attribute, that Address becomes part of the User.addresses collection on that User object:

>>> a2 = Address(email_address="pearl@aol.com", user=u1)
>>> u1.addresses
[Address(id=None, email_address='pearl.krabs@gmail.com'), Address(id=None, email_address='pearl@aol.com')]
We actually made use of the user parameter as a keyword argument in the Address constructor, which is accepted just like any other mapped attribute that was declared on the Address class. It is equivalent to assignment of the Address.user attribute after the fact:

# equivalent effect as a2 = Address(user=u1)
>>> a2.user = u1
Cascading Objects into the Session
We now have a User and two Address objects that are associated in a bidirectional structure in memory, but as noted previously in Inserting Rows using the ORM Unit of Work pattern , these objects are said to be in the transient state until they are associated with a Session object.

We make use of the Session that’s still ongoing, and note that when we apply the Session.add() method to the lead User object, the related Address object also gets added to that same Session:

>>> session.add(u1)
>>> u1 in session
True
>>> a1 in session
True
>>> a2 in session
True
The above behavior, where the Session received a User object, and followed along the User.addresses relationship to locate a related Address object, is known as the save-update cascade and is discussed in detail in the ORM reference documentation at Cascades.

The three objects are now in the pending state; this means they are ready to be the subject of an INSERT operation but this has not yet proceeded; all three objects have no primary key assigned yet, and in addition, the a1 and a2 objects have an attribute called user_id which refers to the Column that has a ForeignKeyConstraint referring to the user_account.id column; these are also None as the objects are not yet associated with a real database row:

>>> print(u1.id)
None
>>> print(a1.user_id)
None
It’s at this stage that we can see the very great utility that the unit of work process provides; recall in the section INSERT usually generates the “values” clause automatically, rows were inserted into the user_account and address tables using some elaborate syntaxes in order to automatically associate the address.user_id columns with those of the user_account rows. Additionally, it was necessary that we emit INSERT for user_account rows first, before those of address, since rows in address are dependent on their parent row in user_account for a value in their user_id column.

When using the Session, all this tedium is handled for us and even the most die-hard SQL purist can benefit from automation of INSERT, UPDATE and DELETE statements. When we Session.commit() the transaction all steps invoke in the correct order, and furthermore the newly generated primary key of the user_account row is applied to the address.user_id column appropriately:

>>> session.commit()
INSERT INTO user_account (name, fullname) VALUES (?, ?)
[...] ('pkrabs', 'Pearl Krabs')
INSERT INTO address (email_address, user_id) VALUES (?, ?) RETURNING id
[... (insertmanyvalues) 1/2 (ordered; batch not supported)] ('pearl.krabs@gmail.com', 6)
INSERT INTO address (email_address, user_id) VALUES (?, ?) RETURNING id
[insertmanyvalues 2/2 (ordered; batch not supported)] ('pearl@aol.com', 6)
COMMIT
Loading Relationships
In the last step, we called Session.commit() which emitted a COMMIT for the transaction, and then per Session.commit.expire_on_commit expired all objects so that they refresh for the next transaction.

When we next access an attribute on these objects, we’ll see the SELECT emitted for the primary attributes of the row, such as when we view the newly generated primary key for the u1 object:

>>> u1.id
BEGIN (implicit)
SELECT user_account.id AS user_account_id, user_account.name AS user_account_name,
user_account.fullname AS user_account_fullname
FROM user_account
WHERE user_account.id = ?
[...] (6,)
6
The u1 User object now has a persistent collection User.addresses that we may also access. As this collection consists of an additional set of rows from the address table, when we access this collection as well we again see a lazy load emitted in order to retrieve the objects:

>>> u1.addresses
SELECT address.id AS address_id, address.email_address AS address_email_address,
address.user_id AS address_user_id
FROM address
WHERE ? = address.user_id
[...] (6,)
[Address(id=4, email_address='pearl.krabs@gmail.com'), Address(id=5, email_address='pearl@aol.com')]
Collections and related attributes in the SQLAlchemy ORM are persistent in memory; once the collection or attribute is populated, SQL is no longer emitted until that collection or attribute is expired. We may access u1.addresses again as well as add or remove items and this will not incur any new SQL calls:

>>> u1.addresses
[Address(id=4, email_address='pearl.krabs@gmail.com'), Address(id=5, email_address='pearl@aol.com')]
While the loading emitted by lazy loading can quickly become expensive if we don’t take explicit steps to optimize it, the network of lazy loading at least is fairly well optimized to not perform redundant work; as the u1.addresses collection was refreshed, per the identity map these are in fact the same Address instances as the a1 and a2 objects we’ve been dealing with already, so we’re done loading all attributes in this particular object graph:

>>> a1
Address(id=4, email_address='pearl.krabs@gmail.com')
>>> a2
Address(id=5, email_address='pearl@aol.com')
The issue of how relationships load, or not, is an entire subject onto itself. Some additional introduction to these concepts is later in this section at Loader Strategies.

Using Relationships in Queries
The previous section introduced the behavior of the relationship() construct when working with instances of a mapped class, above, the u1, a1 and a2 instances of the User and Address classes. In this section, we introduce the behavior of relationship() as it applies to class level behavior of a mapped class, where it serves in several ways to help automate the construction of SQL queries.

Using Relationships to Join
The sections Explicit FROM clauses and JOINs and Setting the ON Clause introduced the usage of the Select.join() and Select.join_from() methods to compose SQL JOIN clauses. In order to describe how to join between tables, these methods either infer the ON clause based on the presence of a single unambiguous ForeignKeyConstraint object within the table metadata structure that links the two tables, or otherwise we may provide an explicit SQL Expression construct that indicates a specific ON clause.

When using ORM entities, an additional mechanism is available to help us set up the ON clause of a join, which is to make use of the relationship() objects that we set up in our user mapping, as was demonstrated at Declaring Mapped Classes. The class-bound attribute corresponding to the relationship() may be passed as the single argument to Select.join(), where it serves to indicate both the right side of the join as well as the ON clause at once:

>>> print(select(Address.email_address).select_from(User).join(User.addresses))
SELECT address.email_address
FROM user_account JOIN address ON user_account.id = address.user_id
The presence of an ORM relationship() on a mapping is not used by Select.join() or Select.join_from() to infer the ON clause if we don’t specify it. This means, if we join from User to Address without an ON clause, it works because of the ForeignKeyConstraint between the two mapped Table objects, not because of the relationship() objects on the User and Address classes:

>>> print(select(Address.email_address).join_from(User, Address))
SELECT address.email_address
FROM user_account JOIN address ON user_account.id = address.user_id
See the section Joins in the ORM Querying Guide for many more examples of how to use Select.join() and Select.join_from() with relationship() constructs.

See also

Joins in the ORM Querying Guide

Relationship WHERE Operators
There are some additional varieties of SQL generation helpers that come with relationship() which are typically useful when building up the WHERE clause of a statement. See the section Relationship WHERE Operators in the ORM Querying Guide.

See also

Relationship WHERE Operators in the ORM Querying Guide

Loader Strategies
In the section Loading Relationships we introduced the concept that when we work with instances of mapped objects, accessing the attributes that are mapped using relationship() in the default case will emit a lazy load when the collection is not populated in order to load the objects that should be present in this collection.

Lazy loading is one of the most famous ORM patterns, and is also the one that is most controversial. When several dozen ORM objects in memory each refer to a handful of unloaded attributes, routine manipulation of these objects can spin off many additional queries that can add up (otherwise known as the N plus one problem), and to make matters worse they are emitted implicitly. These implicit queries may not be noticed, may cause errors when they are attempted after there’s no longer a database transaction available, or when using alternative concurrency patterns such as asyncio, they actually won’t work at all.

At the same time, lazy loading is a vastly popular and useful pattern when it is compatible with the concurrency approach in use and isn’t otherwise causing problems. For these reasons, SQLAlchemy’s ORM places a lot of emphasis on being able to control and optimize this loading behavior.

Above all, the first step in using ORM lazy loading effectively is to test the application, turn on SQL echoing, and watch the SQL that is emitted. If there seem to be lots of redundant SELECT statements that look very much like they could be rolled into one much more efficiently, if there are loads occurring inappropriately for objects that have been detached from their Session, that’s when to look into using loader strategies.

Loader strategies are represented as objects that may be associated with a SELECT statement using the Select.options() method, e.g.:

for user_obj in session.execute(
    select(User).options(selectinload(User.addresses))
).scalars():
    user_obj.addresses  # access addresses collection already loaded
They may be also configured as defaults for a relationship() using the relationship.lazy option, e.g.:

from sqlalchemy.orm import Mapped
from sqlalchemy.orm import relationship


class User(Base):
    __tablename__ = "user_account"

    addresses: Mapped[List["Address"]] = relationship(
        back_populates="user", lazy="selectin"
    )
Each loader strategy object adds some kind of information to the statement that will be used later by the Session when it is deciding how various attributes should be loaded and/or behave when they are accessed.

The sections below will introduce a few of the most prominently used loader strategies.

See also

Two sections in Relationship Loading Techniques:

Configuring Loader Strategies at Mapping Time - details on configuring the strategy on relationship()

Relationship Loading with Loader Options - details on using query-time loader strategies

Selectin Load
The most useful loader in modern SQLAlchemy is the selectinload() loader option. This option solves the most common form of the “N plus one” problem which is that of a set of objects that refer to related collections. selectinload() will ensure that a particular collection for a full series of objects are loaded up front using a single query. It does this using a SELECT form that in most cases can be emitted against the related table alone, without the introduction of JOINs or subqueries, and only queries for those parent objects for which the collection isn’t already loaded. Below we illustrate selectinload() by loading all of the User objects and all of their related Address objects; while we invoke Session.execute() only once, given a select() construct, when the database is accessed, there are in fact two SELECT statements emitted, the second one being to fetch the related Address objects:

>>> from sqlalchemy.orm import selectinload
>>> stmt = select(User).options(selectinload(User.addresses)).order_by(User.id)
>>> for row in session.execute(stmt):
...     print(
...         f"{row.User.name}  ({', '.join(a.email_address for a in row.User.addresses)})"
...     )
SELECT user_account.id, user_account.name, user_account.fullname
FROM user_account ORDER BY user_account.id
[...] ()
SELECT address.user_id AS address_user_id, address.id AS address_id,
address.email_address AS address_email_address
FROM address
WHERE address.user_id IN (?, ?, ?, ?, ?, ?)
[...] (1, 2, 3, 4, 5, 6)
spongebob  (spongebob@sqlalchemy.org)
sandy  (sandy@sqlalchemy.org, sandy@squirrelpower.org)
patrick  ()
squidward  ()
ehkrabs  ()
pkrabs  (pearl.krabs@gmail.com, pearl@aol.com)
See also

Select IN loading - in Relationship Loading Techniques

Joined Load
The joinedload() eager load strategy is the oldest eager loader in SQLAlchemy, which augments the SELECT statement that’s being passed to the database with a JOIN (which may be an outer or an inner join depending on options), which can then load in related objects.

The joinedload() strategy is best suited towards loading related many-to-one objects, as this only requires that additional columns are added to a primary entity row that would be fetched in any case. For greater efficiency, it also accepts an option joinedload.innerjoin so that an inner join instead of an outer join may be used for a case such as below where we know that all Address objects have an associated User:

>>> from sqlalchemy.orm import joinedload
>>> stmt = (
...     select(Address)
...     .options(joinedload(Address.user, innerjoin=True))
...     .order_by(Address.id)
... )
>>> for row in session.execute(stmt):
...     print(f"{row.Address.email_address} {row.Address.user.name}")
SELECT address.id, address.email_address, address.user_id, user_account_1.id AS id_1,
user_account_1.name, user_account_1.fullname
FROM address
JOIN user_account AS user_account_1 ON user_account_1.id = address.user_id
ORDER BY address.id
[...] ()
spongebob@sqlalchemy.org spongebob
sandy@sqlalchemy.org sandy
sandy@squirrelpower.org sandy
pearl.krabs@gmail.com pkrabs
pearl@aol.com pkrabs
joinedload() also works for collections, meaning one-to-many relationships, however it has the effect of multiplying out primary rows per related item in a recursive way that grows the amount of data sent for a result set by orders of magnitude for nested collections and/or larger collections, so its use vs. another option such as selectinload() should be evaluated on a per-case basis.

It’s important to note that the WHERE and ORDER BY criteria of the enclosing Select statement do not target the table rendered by joinedload(). Above, it can be seen in the SQL that an anonymous alias is applied to the user_account table such that is not directly addressable in the query. This concept is discussed in more detail in the section The Zen of Joined Eager Loading.

Tip

It’s important to note that many-to-one eager loads are often not necessary, as the “N plus one” problem is much less prevalent in the common case. When many objects all refer to the same related object, such as many Address objects that each refer to the same User, SQL will be emitted only once for that User object using normal lazy loading. The lazy load routine will look up the related object by primary key in the current Session without emitting any SQL when possible.

See also

Joined Eager Loading - in Relationship Loading Techniques

Explicit Join + Eager load
If we were to load Address rows while joining to the user_account table using a method such as Select.join() to render the JOIN, we could also leverage that JOIN in order to eagerly load the contents of the Address.user attribute on each Address object returned. This is essentially that we are using “joined eager loading” but rendering the JOIN ourselves. This common use case is achieved by using the contains_eager() option. This option is very similar to joinedload(), except that it assumes we have set up the JOIN ourselves, and it instead only indicates that additional columns in the COLUMNS clause should be loaded into related attributes on each returned object, for example:

>>> from sqlalchemy.orm import contains_eager
>>> stmt = (
...     select(Address)
...     .join(Address.user)
...     .where(User.name == "pkrabs")
...     .options(contains_eager(Address.user))
...     .order_by(Address.id)
... )
>>> for row in session.execute(stmt):
...     print(f"{row.Address.email_address} {row.Address.user.name}")
SELECT user_account.id, user_account.name, user_account.fullname,
address.id AS id_1, address.email_address, address.user_id
FROM address JOIN user_account ON user_account.id = address.user_id
WHERE user_account.name = ? ORDER BY address.id
[...] ('pkrabs',)
pearl.krabs@gmail.com pkrabs
pearl@aol.com pkrabs
Above, we both filtered the rows on user_account.name and also loaded rows from user_account into the Address.user attribute of the returned rows. If we had applied joinedload() separately, we would get a SQL query that unnecessarily joins twice:

>>> stmt = (
...     select(Address)
...     .join(Address.user)
...     .where(User.name == "pkrabs")
...     .options(joinedload(Address.user))
...     .order_by(Address.id)
... )
>>> print(stmt)  # SELECT has a JOIN and LEFT OUTER JOIN unnecessarily
SELECT address.id, address.email_address, address.user_id,
user_account_1.id AS id_1, user_account_1.name, user_account_1.fullname
FROM address JOIN user_account ON user_account.id = address.user_id
LEFT OUTER JOIN user_account AS user_account_1 ON user_account_1.id = address.user_id
WHERE user_account.name = :name_1 ORDER BY address.id
See also

Two sections in Relationship Loading Techniques:

The Zen of Joined Eager Loading - describes the above problem in detail

Routing Explicit Joins/Statements into Eagerly Loaded Collections - using contains_eager()

Raiseload
One additional loader strategy worth mentioning is raiseload(). This option is used to completely block an application from having the N plus one problem at all by causing what would normally be a lazy load to raise an error instead. It has two variants that are controlled via the raiseload.sql_only option to block either lazy loads that require SQL, versus all “load” operations including those which only need to consult the current Session.

One way to use raiseload() is to configure it on relationship() itself, by setting relationship.lazy to the value "raise_on_sql", so that for a particular mapping, a certain relationship will never try to emit SQL:

>>> from sqlalchemy.orm import Mapped
>>> from sqlalchemy.orm import relationship


>>> class User(Base):
...     __tablename__ = "user_account"
...     id: Mapped[int] = mapped_column(primary_key=True)
...     addresses: Mapped[List["Address"]] = relationship(
...         back_populates="user", lazy="raise_on_sql"
...     )


>>> class Address(Base):
...     __tablename__ = "address"
...     id: Mapped[int] = mapped_column(primary_key=True)
...     user_id: Mapped[int] = mapped_column(ForeignKey("user_account.id"))
...     user: Mapped["User"] = relationship(back_populates="addresses", lazy="raise_on_sql")
Using such a mapping, the application is blocked from lazy loading, indicating that a particular query would need to specify a loader strategy:

>>> u1 = session.execute(select(User)).scalars().first()
SELECT user_account.id FROM user_account
[...] ()
>>> u1.addresses
Traceback (most recent call last):
...
sqlalchemy.exc.InvalidRequestError: 'User.addresses' is not available due to lazy='raise_on_sql'
The exception would indicate that this collection should be loaded up front instead:

>>> u1 = (
...     session.execute(select(User).options(selectinload(User.addresses)))
...     .scalars()
...     .first()
... )
SELECT user_account.id
FROM user_account
[...] ()
SELECT address.user_id AS address_user_id, address.id AS address_id
FROM address
WHERE address.user_id IN (?, ?, ?, ?, ?, ?)
[...] (1, 2, 3, 4, 5, 6)
The lazy="raise_on_sql" option tries to be smart about many-to-one relationships as well; above, if the Address.user attribute of an Address object were not loaded, but that User object were locally present in the same Session, the “raiseload” strategy would not raise an error.

See also

Preventing unwanted lazy loads using raiseload - in Relationship Loading Techniques

SQLAlchemy 1.4 / 2.0 Tutorial

Next Tutorial Section: Further Reading

Basic Relationship Patterns
A quick walkthrough of the basic relational patterns, which in this section are illustrated using Declarative style mappings based on the use of the Mapped annotation type.

The setup for each of the following sections is as follows:

from __future__ import annotations
from typing import List

from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import relationship


class Base(DeclarativeBase):
    pass
Declarative vs. Imperative Forms
As SQLAlchemy has evolved, different ORM configurational styles have emerged. For examples in this section and others that use annotated Declarative mappings with Mapped, the corresponding non-annotated form should use the desired class, or string class name, as the first argument passed to relationship(). The example below illustrates the form used in this document, which is a fully Declarative example using PEP 484 annotations, where the relationship() construct is also deriving the target class and collection type from the Mapped annotation, which is the most modern form of SQLAlchemy Declarative mapping:

class Parent(Base):
    __tablename__ = "parent_table"

    id: Mapped[int] = mapped_column(primary_key=True)
    children: Mapped[List["Child"]] = relationship(back_populates="parent")


class Child(Base):
    __tablename__ = "child_table"

    id: Mapped[int] = mapped_column(primary_key=True)
    parent_id: Mapped[int] = mapped_column(ForeignKey("parent_table.id"))
    parent: Mapped["Parent"] = relationship(back_populates="children")
In contrast, using a Declarative mapping without annotations is the more “classic” form of mapping, where relationship() requires all parameters passed to it directly, as in the example below:

class Parent(Base):
    __tablename__ = "parent_table"

    id = mapped_column(Integer, primary_key=True)
    children = relationship("Child", back_populates="parent")


class Child(Base):
    __tablename__ = "child_table"

    id = mapped_column(Integer, primary_key=True)
    parent_id = mapped_column(ForeignKey("parent_table.id"))
    parent = relationship("Parent", back_populates="children")
Finally, using Imperative Mapping, which is SQLAlchemy’s original mapping form before Declarative was made (which nonetheless remains preferred by a vocal minority of users), the above configuration looks like:

registry.map_imperatively(
    Parent,
    parent_table,
    properties={"children": relationship("Child", back_populates="parent")},
)

registry.map_imperatively(
    Child,
    child_table,
    properties={"parent": relationship("Parent", back_populates="children")},
)
Additionally, the default collection style for non-annotated mappings is list. To use a set or other collection without annotations, indicate it using the relationship.collection_class parameter:

class Parent(Base):
    __tablename__ = "parent_table"

    id = mapped_column(Integer, primary_key=True)
    children = relationship("Child", collection_class=set, ...)
Detail on collection configuration for relationship() is at Customizing Collection Access.

Additional differences between annotated and non-annotated / imperative styles will be noted as needed.

One To Many
A one to many relationship places a foreign key on the child table referencing the parent. relationship() is then specified on the parent, as referencing a collection of items represented by the child:

class Parent(Base):
    __tablename__ = "parent_table"

    id: Mapped[int] = mapped_column(primary_key=True)
    children: Mapped[List["Child"]] = relationship()


class Child(Base):
    __tablename__ = "child_table"

    id: Mapped[int] = mapped_column(primary_key=True)
    parent_id: Mapped[int] = mapped_column(ForeignKey("parent_table.id"))
To establish a bidirectional relationship in one-to-many, where the “reverse” side is a many to one, specify an additional relationship() and connect the two using the relationship.back_populates parameter, using the attribute name of each relationship() as the value for relationship.back_populates on the other:

class Parent(Base):
    __tablename__ = "parent_table"

    id: Mapped[int] = mapped_column(primary_key=True)
    children: Mapped[List["Child"]] = relationship(back_populates="parent")


class Child(Base):
    __tablename__ = "child_table"

    id: Mapped[int] = mapped_column(primary_key=True)
    parent_id: Mapped[int] = mapped_column(ForeignKey("parent_table.id"))
    parent: Mapped["Parent"] = relationship(back_populates="children")
Child will get a parent attribute with many-to-one semantics.

Using Sets, Lists, or other Collection Types for One To Many
Using annotated Declarative mappings, the type of collection used for the relationship() is derived from the collection type passed to the Mapped container type. The example from the previous section may be written to use a set rather than a list for the Parent.children collection using Mapped[Set["Child"]]:

class Parent(Base):
    __tablename__ = "parent_table"

    id: Mapped[int] = mapped_column(primary_key=True)
    children: Mapped[Set["Child"]] = relationship(back_populates="parent")
When using non-annotated forms including imperative mappings, the Python class to use as a collection may be passed using the relationship.collection_class parameter.

See also

Customizing Collection Access - contains further detail on collection configuration including some techniques to map relationship() to dictionaries.

Configuring Delete Behavior for One to Many
It is often the case that all Child objects should be deleted when their owning Parent is deleted. To configure this behavior, the delete cascade option described at delete is used. An additional option is that a Child object can itself be deleted when it is deassociated from its parent. This behavior is described at delete-orphan.

See also

delete

Using foreign key ON DELETE cascade with ORM relationships

delete-orphan

Many To One
Many to one places a foreign key in the parent table referencing the child. relationship() is declared on the parent, where a new scalar-holding attribute will be created:

class Parent(Base):
    __tablename__ = "parent_table"

    id: Mapped[int] = mapped_column(primary_key=True)
    child_id: Mapped[int] = mapped_column(ForeignKey("child_table.id"))
    child: Mapped["Child"] = relationship()


class Child(Base):
    __tablename__ = "child_table"

    id: Mapped[int] = mapped_column(primary_key=True)
The above example shows a many-to-one relationship that assumes non-nullable behavior; the next section, Nullable Many-to-One, illustrates a nullable version.

Bidirectional behavior is achieved by adding a second relationship() and applying the relationship.back_populates parameter in both directions, using the attribute name of each relationship() as the value for relationship.back_populates on the other:

class Parent(Base):
    __tablename__ = "parent_table"

    id: Mapped[int] = mapped_column(primary_key=True)
    child_id: Mapped[int] = mapped_column(ForeignKey("child_table.id"))
    child: Mapped["Child"] = relationship(back_populates="parents")


class Child(Base):
    __tablename__ = "child_table"

    id: Mapped[int] = mapped_column(primary_key=True)
    parents: Mapped[List["Parent"]] = relationship(back_populates="child")
Nullable Many-to-One
In the preceding example, the Parent.child relationship is not typed as allowing None; this follows from the Parent.child_id column itself not being nullable, as it is typed with Mapped[int]. If we wanted Parent.child to be a nullable many-to-one, we can set both Parent.child_id and Parent.child to be Optional[], in which case the configuration would look like:

from typing import Optional


class Parent(Base):
    __tablename__ = "parent_table"

    id: Mapped[int] = mapped_column(primary_key=True)
    child_id: Mapped[Optional[int]] = mapped_column(ForeignKey("child_table.id"))
    child: Mapped[Optional["Child"]] = relationship(back_populates="parents")


class Child(Base):
    __tablename__ = "child_table"

    id: Mapped[int] = mapped_column(primary_key=True)
    parents: Mapped[List["Parent"]] = relationship(back_populates="child")
Above, the column for Parent.child_id will be created in DDL to allow NULL values. When using mapped_column() with explicit typing declarations, the specification of child_id: Mapped[Optional[int]] is equivalent to setting Column.nullable to True on the Column, whereas child_id: Mapped[int] is equivalent to setting it to False. See mapped_column() derives the datatype and nullability from the Mapped annotation for background on this behavior.

Tip

If using Python 3.10 or greater, PEP 604 syntax is more convenient to indicate optional types using | None, which when combined with PEP 563 postponed annotation evaluation so that string-quoted types aren’t required, would look like:

from __future__ import annotations


class Parent(Base):
    __tablename__ = "parent_table"

    id: Mapped[int] = mapped_column(primary_key=True)
    child_id: Mapped[int | None] = mapped_column(ForeignKey("child_table.id"))
    child: Mapped[Child | None] = relationship(back_populates="parents")


class Child(Base):
    __tablename__ = "child_table"

    id: Mapped[int] = mapped_column(primary_key=True)
    parents: Mapped[List[Parent]] = relationship(back_populates="child")
One To One
One To One is essentially a One To Many relationship from a foreign key perspective, but indicates that there will only be one row at any time that refers to a particular parent row.

When using annotated mappings with Mapped, the “one-to-one” convention is achieved by applying a non-collection type to the Mapped annotation on both sides of the relationship, which will imply to the ORM that a collection should not be used on either side, as in the example below:

class Parent(Base):
    __tablename__ = "parent_table"

    id: Mapped[int] = mapped_column(primary_key=True)
    child: Mapped["Child"] = relationship(back_populates="parent")


class Child(Base):
    __tablename__ = "child_table"

    id: Mapped[int] = mapped_column(primary_key=True)
    parent_id: Mapped[int] = mapped_column(ForeignKey("parent_table.id"))
    parent: Mapped["Parent"] = relationship(back_populates="child")
Above, when we load a Parent object, the Parent.child attribute will refer to a single Child object rather than a collection. If we replace the value of Parent.child with a new Child object, the ORM’s unit of work process will replace the previous Child row with the new one, setting the previous child.parent_id column to NULL by default unless there are specific cascade behaviors set up.

Tip

As mentioned previously, the ORM considers the “one-to-one” pattern as a convention, where it makes the assumption that when it loads the Parent.child attribute on a Parent object, it will get only one row back. If more than one row is returned, the ORM will emit a warning.

However, the Child.parent side of the above relationship remains as a “many-to-one” relationship. By itself, it will not detect assignment of more than one Child, unless the relationship.single_parent parameter is set, which may be useful:

class Child(Base):
    __tablename__ = "child_table"

    id: Mapped[int] = mapped_column(primary_key=True)
    parent_id: Mapped[int] = mapped_column(ForeignKey("parent_table.id"))
    parent: Mapped["Parent"] = relationship(back_populates="child", single_parent=True)
Outside of setting this parameter, the “one-to-many” side (which here is one-to-one by convention) will also not reliably detect if more than one Child is associated with a single Parent, such as in the case where the multiple Child objects are pending and not database-persistent.

Whether or not relationship.single_parent is used, it is recommended that the database schema include a unique constraint to indicate that the Child.parent_id column should be unique, to ensure at the database level that only one Child row may refer to a particular Parent row at a time (see Declarative Table Configuration for background on the __table_args__ tuple syntax):

from sqlalchemy import UniqueConstraint


class Child(Base):
    __tablename__ = "child_table"

    id: Mapped[int] = mapped_column(primary_key=True)
    parent_id: Mapped[int] = mapped_column(ForeignKey("parent_table.id"))
    parent: Mapped["Parent"] = relationship(back_populates="child")

    __table_args__ = (UniqueConstraint("parent_id"),)
New in version 2.0: The relationship() construct can derive the effective value of the relationship.uselist parameter from a given Mapped annotation.

Setting uselist=False for non-annotated configurations
When using relationship() without the benefit of Mapped annotations, the one-to-one pattern can be enabled using the relationship.uselist parameter set to False on what would normally be the “many” side, illustrated in a non-annotated Declarative configuration below:

class Parent(Base):
    __tablename__ = "parent_table"

    id = mapped_column(Integer, primary_key=True)
    child = relationship("Child", uselist=False, back_populates="parent")


class Child(Base):
    __tablename__ = "child_table"

    id = mapped_column(Integer, primary_key=True)
    parent_id = mapped_column(ForeignKey("parent_table.id"))
    parent = relationship("Parent", back_populates="child")
Many To Many
Many to Many adds an association table between two classes. The association table is nearly always given as a Core Table object or other Core selectable such as a Join object, and is indicated by the relationship.secondary argument to relationship(). Usually, the Table uses the MetaData object associated with the declarative base class, so that the ForeignKey directives can locate the remote tables with which to link:

from __future__ import annotations

from sqlalchemy import Column
from sqlalchemy import Table
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import relationship


class Base(DeclarativeBase):
    pass


# note for a Core table, we use the sqlalchemy.Column construct,
# not sqlalchemy.orm.mapped_column
association_table = Table(
    "association_table",
    Base.metadata,
    Column("left_id", ForeignKey("left_table.id")),
    Column("right_id", ForeignKey("right_table.id")),
)


class Parent(Base):
    __tablename__ = "left_table"

    id: Mapped[int] = mapped_column(primary_key=True)
    children: Mapped[List[Child]] = relationship(secondary=association_table)


class Child(Base):
    __tablename__ = "right_table"

    id: Mapped[int] = mapped_column(primary_key=True)
Tip

The “association table” above has foreign key constraints established that refer to the two entity tables on either side of the relationship. The data type of each of association.left_id and association.right_id is normally inferred from that of the referenced table and may be omitted. It is also recommended, though not in any way required by SQLAlchemy, that the columns which refer to the two entity tables are established within either a unique constraint or more commonly as the primary key constraint; this ensures that duplicate rows won’t be persisted within the table regardless of issues on the application side:

association_table = Table(
    "association_table",
    Base.metadata,
    Column("left_id", ForeignKey("left_table.id"), primary_key=True),
    Column("right_id", ForeignKey("right_table.id"), primary_key=True),
)
Setting Bi-Directional Many-to-many
For a bidirectional relationship, both sides of the relationship contain a collection. Specify using relationship.back_populates, and for each relationship() specify the common association table:

from __future__ import annotations

from sqlalchemy import Column
from sqlalchemy import Table
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import relationship


class Base(DeclarativeBase):
    pass


association_table = Table(
    "association_table",
    Base.metadata,
    Column("left_id", ForeignKey("left_table.id"), primary_key=True),
    Column("right_id", ForeignKey("right_table.id"), primary_key=True),
)


class Parent(Base):
    __tablename__ = "left_table"

    id: Mapped[int] = mapped_column(primary_key=True)
    children: Mapped[List[Child]] = relationship(
        secondary=association_table, back_populates="parents"
    )


class Child(Base):
    __tablename__ = "right_table"

    id: Mapped[int] = mapped_column(primary_key=True)
    parents: Mapped[List[Parent]] = relationship(
        secondary=association_table, back_populates="children"
    )
Using a late-evaluated form for the “secondary” argument
The relationship.secondary parameter of relationship() also accepts two different “late evaluated” forms, including string table name as well as lambda callable. See the section Using a late-evaluated form for the “secondary” argument of many-to-many for background and examples.

Using Sets, Lists, or other Collection Types for Many To Many
Configuration of collections for a Many to Many relationship is identical to that of One To Many, as described at Using Sets, Lists, or other Collection Types for One To Many. For an annotated mapping using Mapped, the collection can be indicated by the type of collection used within the Mapped generic class, such as set:

class Parent(Base):
    __tablename__ = "left_table"

    id: Mapped[int] = mapped_column(primary_key=True)
    children: Mapped[Set["Child"]] = relationship(secondary=association_table)
When using non-annotated forms including imperative mappings, as is the case with one-to-many, the Python class to use as a collection may be passed using the relationship.collection_class parameter.

See also

Customizing Collection Access - contains further detail on collection configuration including some techniques to map relationship() to dictionaries.

Deleting Rows from the Many to Many Table
A behavior which is unique to the relationship.secondary argument to relationship() is that the Table which is specified here is automatically subject to INSERT and DELETE statements, as objects are added or removed from the collection. There is no need to delete from this table manually. The act of removing a record from the collection will have the effect of the row being deleted on flush:

# row will be deleted from the "secondary" table
# automatically
myparent.children.remove(somechild)
A question which often arises is how the row in the “secondary” table can be deleted when the child object is handed directly to Session.delete():

session.delete(somechild)
There are several possibilities here:

If there is a relationship() from Parent to Child, but there is not a reverse-relationship that links a particular Child to each Parent, SQLAlchemy will not have any awareness that when deleting this particular Child object, it needs to maintain the “secondary” table that links it to the Parent. No delete of the “secondary” table will occur.

If there is a relationship that links a particular Child to each Parent, suppose it’s called Child.parents, SQLAlchemy by default will load in the Child.parents collection to locate all Parent objects, and remove each row from the “secondary” table which establishes this link. Note that this relationship does not need to be bidirectional; SQLAlchemy is strictly looking at every relationship() associated with the Child object being deleted.

A higher performing option here is to use ON DELETE CASCADE directives with the foreign keys used by the database. Assuming the database supports this feature, the database itself can be made to automatically delete rows in the “secondary” table as referencing rows in “child” are deleted. SQLAlchemy can be instructed to forego actively loading in the Child.parents collection in this case using the relationship.passive_deletes directive on relationship(); see Using foreign key ON DELETE cascade with ORM relationships for more details on this.

Note again, these behaviors are only relevant to the relationship.secondary option used with relationship(). If dealing with association tables that are mapped explicitly and are not present in the relationship.secondary option of a relevant relationship(), cascade rules can be used instead to automatically delete entities in reaction to a related entity being deleted - see Cascades for information on this feature.

See also

Using delete cascade with many-to-many relationships

Using foreign key ON DELETE with many-to-many relationships

Association Object
The association object pattern is a variant on many-to-many: it’s used when an association table contains additional columns beyond those which are foreign keys to the parent and child (or left and right) tables, columns which are most ideally mapped to their own ORM mapped class. This mapped class is mapped against the Table that would otherwise be noted as relationship.secondary when using the many-to-many pattern.

In the association object pattern, the relationship.secondary parameter is not used; instead, a class is mapped directly to the association table. Two individual relationship() constructs then link first the parent side to the mapped association class via one to many, and then the mapped association class to the child side via many-to-one, to form a uni-directional association object relationship from parent, to association, to child. For a bi-directional relationship, four relationship() constructs are used to link the mapped association class to both parent and child in both directions.

The example below illustrates a new class Association which maps to the Table named association; this table now includes an additional column called extra_data, which is a string value that is stored along with each association between Parent and Child. By mapping the table to an explicit class, rudimental access from Parent to Child makes explicit use of Association:

from typing import Optional

from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import relationship


class Base(DeclarativeBase):
    pass


class Association(Base):
    __tablename__ = "association_table"
    left_id: Mapped[int] = mapped_column(ForeignKey("left_table.id"), primary_key=True)
    right_id: Mapped[int] = mapped_column(
        ForeignKey("right_table.id"), primary_key=True
    )
    extra_data: Mapped[Optional[str]]
    child: Mapped["Child"] = relationship()


class Parent(Base):
    __tablename__ = "left_table"
    id: Mapped[int] = mapped_column(primary_key=True)
    children: Mapped[List["Association"]] = relationship()


class Child(Base):
    __tablename__ = "right_table"
    id: Mapped[int] = mapped_column(primary_key=True)
To illustrate the bi-directional version, we add two more relationship() constructs, linked to the existing ones using relationship.back_populates:

from typing import Optional

from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import relationship


class Base(DeclarativeBase):
    pass


class Association(Base):
    __tablename__ = "association_table"
    left_id: Mapped[int] = mapped_column(ForeignKey("left_table.id"), primary_key=True)
    right_id: Mapped[int] = mapped_column(
        ForeignKey("right_table.id"), primary_key=True
    )
    extra_data: Mapped[Optional[str]]
    child: Mapped["Child"] = relationship(back_populates="parents")
    parent: Mapped["Parent"] = relationship(back_populates="children")


class Parent(Base):
    __tablename__ = "left_table"
    id: Mapped[int] = mapped_column(primary_key=True)
    children: Mapped[List["Association"]] = relationship(back_populates="parent")


class Child(Base):
    __tablename__ = "right_table"
    id: Mapped[int] = mapped_column(primary_key=True)
    parents: Mapped[List["Association"]] = relationship(back_populates="child")
Working with the association pattern in its direct form requires that child objects are associated with an association instance before being appended to the parent; similarly, access from parent to child goes through the association object:

# create parent, append a child via association
p = Parent()
a = Association(extra_data="some data")
a.child = Child()
p.children.append(a)

# iterate through child objects via association, including association
# attributes
for assoc in p.children:
    print(assoc.extra_data)
    print(assoc.child)
To enhance the association object pattern such that direct access to the Association object is optional, SQLAlchemy provides the Association Proxy extension. This extension allows the configuration of attributes which will access two “hops” with a single access, one “hop” to the associated object, and a second to a target attribute.

See also

Association Proxy - allows direct “many to many” style access between parent and child for a three-class association object mapping.

Warning

Avoid mixing the association object pattern with the many-to-many pattern directly, as this produces conditions where data may be read and written in an inconsistent fashion without special steps; the association proxy is typically used to provide more succinct access. For more detailed background on the caveats introduced by this combination, see the next section Combining Association Object with Many-to-Many Access Patterns.

Combining Association Object with Many-to-Many Access Patterns
As mentioned in the previous section, the association object pattern does not automatically integrate with usage of the many-to-many pattern against the same tables/columns at the same time. From this it follows that read operations may return conflicting data and write operations may also attempt to flush conflicting changes, causing either integrity errors or unexpected inserts or deletes.

To illustrate, the example below configures a bidirectional many-to-many relationship between Parent and Child via Parent.children and Child.parents. At the same time, an association object relationship is also configured, between Parent.child_associations -> Association.child and Child.parent_associations -> Association.parent:

from typing import Optional

from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import relationship


class Base(DeclarativeBase):
    pass


class Association(Base):
    __tablename__ = "association_table"

    left_id: Mapped[int] = mapped_column(ForeignKey("left_table.id"), primary_key=True)
    right_id: Mapped[int] = mapped_column(
        ForeignKey("right_table.id"), primary_key=True
    )
    extra_data: Mapped[Optional[str]]

    # association between Assocation -> Child
    child: Mapped["Child"] = relationship(back_populates="parent_associations")

    # association between Assocation -> Parent
    parent: Mapped["Parent"] = relationship(back_populates="child_associations")


class Parent(Base):
    __tablename__ = "left_table"

    id: Mapped[int] = mapped_column(primary_key=True)

    # many-to-many relationship to Child, bypassing the `Association` class
    children: Mapped[List["Child"]] = relationship(
        secondary="association_table", back_populates="parents"
    )

    # association between Parent -> Association -> Child
    child_associations: Mapped[List["Association"]] = relationship(
        back_populates="parent"
    )


class Child(Base):
    __tablename__ = "right_table"

    id: Mapped[int] = mapped_column(primary_key=True)

    # many-to-many relationship to Parent, bypassing the `Association` class
    parents: Mapped[List["Parent"]] = relationship(
        secondary="association_table", back_populates="children"
    )

    # association between Child -> Association -> Parent
    parent_associations: Mapped[List["Association"]] = relationship(
        back_populates="child"
    )
When using this ORM model to make changes, changes made to Parent.children will not be coordinated with changes made to Parent.child_associations or Child.parent_associations in Python; while all of these relationships will continue to function normally by themselves, changes on one will not show up in another until the Session is expired, which normally occurs automatically after Session.commit().

Additionally, if conflicting changes are made, such as adding a new Association object while also appending the same related Child to Parent.children, this will raise integrity errors when the unit of work flush process proceeds, as in the example below:

p1 = Parent()
c1 = Child()
p1.children.append(c1)

# redundant, will cause a duplicate INSERT on Association
p1.child_associations.append(Association(child=c1))
Appending Child to Parent.children directly also implies the creation of rows in the association table without indicating any value for the association.extra_data column, which will receive NULL for its value.

It’s fine to use a mapping like the above if you know what you’re doing; there may be good reason to use many-to-many relationships in the case where use of the “association object” pattern is infrequent, which is that it’s easier to load relationships along a single many-to-many relationship, which can also optimize slightly better how the “secondary” table is used in SQL statements, compared to how two separate relationships to an explicit association class is used. It’s at least a good idea to apply the relationship.viewonly parameter to the “secondary” relationship to avoid the issue of conflicting changes occurring, as well as preventing NULL being written to the additional association columns, as below:

class Parent(Base):
    __tablename__ = "left_table"

    id: Mapped[int] = mapped_column(primary_key=True)

    # many-to-many relationship to Child, bypassing the `Association` class
    children: Mapped[List["Child"]] = relationship(
        secondary="association_table", back_populates="parents", viewonly=True
    )

    # association between Parent -> Association -> Child
    child_associations: Mapped[List["Association"]] = relationship(
        back_populates="parent"
    )


class Child(Base):
    __tablename__ = "right_table"

    id: Mapped[int] = mapped_column(primary_key=True)

    # many-to-many relationship to Parent, bypassing the `Association` class
    parents: Mapped[List["Parent"]] = relationship(
        secondary="association_table", back_populates="children", viewonly=True
    )

    # association between Child -> Association -> Parent
    parent_associations: Mapped[List["Association"]] = relationship(
        back_populates="child"
    )
The above mapping will not write any changes to Parent.children or Child.parents to the database, preventing conflicting writes. However, reads of Parent.children or Child.parents will not necessarily match the data that’s read from Parent.child_associations or Child.parent_associations, if changes are being made to these collections within the same transaction or Session as where the viewonly collections are being read. If use of the association object relationships is infrequent and is carefully organized against code that accesses the many-to-many collections to avoid stale reads (in extreme cases, making direct use of Session.expire() to cause collections to be refreshed within the current transaction), the pattern may be feasible.

A popular alternative to the above pattern is one where the direct many-to-many Parent.children and Child.parents relationships are replaced with an extension that will transparently proxy through the Association class, while keeping everything consistent from the ORM’s point of view. This extension is known as the Association Proxy.

See also

Association Proxy - allows direct “many to many” style access between parent and child for a three-class association object mapping.

Late-Evaluation of Relationship Arguments
Most of the examples in the preceding sections illustrate mappings where the various relationship() constructs refer to their target classes using a string name, rather than the class itself, such as when using Mapped, a forward reference is generated that exists at runtime only as a string:

class Parent(Base):
    # ...

    children: Mapped[List["Child"]] = relationship(back_populates="parent")


class Child(Base):
    # ...

    parent: Mapped["Parent"] = relationship(back_populates="children")
Similarly, when using non-annotated forms such as non-annotated Declarative or Imperative mappings, a string name is also supported directly by the relationship() construct:

registry.map_imperatively(
    Parent,
    parent_table,
    properties={"children": relationship("Child", back_populates="parent")},
)

registry.map_imperatively(
    Child,
    child_table,
    properties={"parent": relationship("Parent", back_populates="children")},
)
These string names are resolved into classes in the mapper resolution stage, which is an internal process that occurs typically after all mappings have been defined and is normally triggered by the first usage of the mappings themselves. The registry object is the container where these names are stored and resolved to the mapped classes to which they refer.

In addition to the main class argument for relationship(), other arguments which depend upon the columns present on an as-yet undefined class may also be specified either as Python functions, or more commonly as strings. For most of these arguments except that of the main argument, string inputs are evaluated as Python expressions using Python’s built-in eval() function, as they are intended to receive complete SQL expressions.

Warning

As the Python eval() function is used to interpret the late-evaluated string arguments passed to relationship() mapper configuration construct, these arguments should not be repurposed such that they would receive untrusted user input; eval() is not secure against untrusted user input.

The full namespace available within this evaluation includes all classes mapped for this declarative base, as well as the contents of the sqlalchemy package, including expression functions like desc() and sqlalchemy.sql.functions.func:

class Parent(Base):
    # ...

    children: Mapped[List["Child"]] = relationship(
        order_by="desc(Child.email_address)",
        primaryjoin="Parent.id == Child.parent_id",
    )
For the case where more than one module contains a class of the same name, string class names can also be specified as module-qualified paths within any of these string expressions:

class Parent(Base):
    # ...

    children: Mapped[List["myapp.mymodel.Child"]] = relationship(
        order_by="desc(myapp.mymodel.Child.email_address)",
        primaryjoin="myapp.mymodel.Parent.id == myapp.mymodel.Child.parent_id",
    )
In an example like the above, the string passed to Mapped can be disambiguated from a specific class argument by passing the class location string directly to relationship.argument as well. Below illustrates a typing-only import for Child, combined with a runtime specifier for the target class that will search for the correct name within the registry:

import typing

if typing.TYPE_CHECKING:
    from myapp.mymodel import Child


class Parent(Base):
    # ...

    children: Mapped[List["Child"]] = relationship(
        "myapp.mymodel.Child",
        order_by="desc(myapp.mymodel.Child.email_address)",
        primaryjoin="myapp.mymodel.Parent.id == myapp.mymodel.Child.parent_id",
    )
The qualified path can be any partial path that removes ambiguity between the names. For example, to disambiguate between myapp.model1.Child and myapp.model2.Child, we can specify model1.Child or model2.Child:

class Parent(Base):
    # ...

    children: Mapped[List["Child"]] = relationship(
        "model1.Child",
        order_by="desc(mymodel1.Child.email_address)",
        primaryjoin="Parent.id == model1.Child.parent_id",
    )
The relationship() construct also accepts Python functions or lambdas as input for these arguments. A Python functional approach might look like the following:

import typing

from sqlalchemy import desc

if typing.TYPE_CHECKING:
    from myapplication import Child


def _resolve_child_model():
    from myapplication import Child

    return Child


class Parent(Base):
    # ...

    children: Mapped[List["Child"]] = relationship(
        _resolve_child_model,
        order_by=lambda: desc(_resolve_child_model().email_address),
        primaryjoin=lambda: Parent.id == _resolve_child_model().parent_id,
    )
The full list of parameters which accept Python functions/lambdas or strings that will be passed to eval() are:

relationship.order_by

relationship.primaryjoin

relationship.secondaryjoin

relationship.secondary

relationship.remote_side

relationship.foreign_keys

relationship._user_defined_foreign_keys

Warning

As stated previously, the above parameters to relationship() are evaluated as Python code expressions using eval(). DO NOT PASS UNTRUSTED INPUT TO THESE ARGUMENTS.

Adding Relationships to Mapped Classes After Declaration
It should also be noted that in a similar way as described at Appending additional columns to an existing Declarative mapped class, any MapperProperty construct can be added to a declarative base mapping at any time (noting that annotated forms are not supported in this context). If we wanted to implement this relationship() after the Address class were available, we could also apply it afterwards:

# first, module A, where Child has not been created yet,
# we create a Parent class which knows nothing about Child


class Parent(Base): ...


# ... later, in Module B, which is imported after module A:


class Child(Base): ...


from module_a import Parent

# assign the User.addresses relationship as a class variable.  The
# declarative base class will intercept this and map the relationship.
Parent.children = relationship(Child, primaryjoin=Child.parent_id == Parent.id)
As is the case for ORM mapped columns, there’s no capability for the Mapped annotation type to take part in this operation; therefore, the related class must be specified directly within the relationship() construct, either as the class itself, the string name of the class, or a callable function that returns a reference to the target class.

Note

As is the case for ORM mapped columns, assignment of mapped properties to an already mapped class will only function correctly if the “declarative base” class is used, meaning the user-defined subclass of DeclarativeBase or the dynamically generated class returned by declarative_base() or registry.generate_base(). This “base” class includes a Python metaclass which implements a special __setattr__() method that intercepts these operations.

Runtime assignment of class-mapped attributes to a mapped class will not work if the class is mapped using decorators like registry.mapped() or imperative functions like registry.map_imperatively().

Using a late-evaluated form for the “secondary” argument of many-to-many
Many-to-many relationships make use of the relationship.secondary parameter, which ordinarily indicates a reference to a typically non-mapped Table object or other Core selectable object. Late evaluation using a lambda callable is typical.

For the example given at Many To Many, if we assumed that the association_table Table object would be defined at a point later on in the module than the mapped class itself, we may write the relationship() using a lambda as:

class Parent(Base):
    __tablename__ = "left_table"

    id: Mapped[int] = mapped_column(primary_key=True)
    children: Mapped[List["Child"]] = relationship(
        "Child", secondary=lambda: association_table
    )
As a shortcut for table names that are also valid Python identifiers, the relationship.secondary parameter may also be passed as a string, where resolution works by evaluation of the string as a Python expression, with simple identifier names linked to same-named Table objects that are present in the same MetaData collection referenced by the current registry.

In the example below, the expression "association_table" is evaluated as a variable named “association_table” that is resolved against the table names within the MetaData collection:

class Parent(Base):
    __tablename__ = "left_table"

    id: Mapped[int] = mapped_column(primary_key=True)
    children: Mapped[List["Child"]] = relationship(secondary="association_table")
Note

When passed as a string, the name passed to relationship.secondary must be a valid Python identifier starting with a letter and containing only alphanumeric characters or underscores. Other characters such as dashes etc. will be interpreted as Python operators which will not resolve to the name given. Please consider using lambda expressions rather than strings for improved clarity.

Warning

When passed as a string, relationship.secondary argument is interpreted using Python’s eval() function, even though it’s typically the name of a table. DO NOT PASS UNTRUSTED INPUT TO THIS STRING.

Previous: Relationship Configuration Next: Adjacency List Relationships
© Copyright 2007-2025, the SQLAlchemy authors and contributors.
flambé! the dragon and The Alchemist image designs created and generously donated by Rotem Yaari.

Adjacency List Relationships
The adjacency list pattern is a common relational pattern whereby a table contains a foreign key reference to itself, in other words is a self referential relationship. This is the most common way to represent hierarchical data in flat tables. Other methods include nested sets, sometimes called “modified preorder”, as well as materialized path. Despite the appeal that modified preorder has when evaluated for its fluency within SQL queries, the adjacency list model is probably the most appropriate pattern for the large majority of hierarchical storage needs, for reasons of concurrency, reduced complexity, and that modified preorder has little advantage over an application which can fully load subtrees into the application space.

See also

This section details the single-table version of a self-referential relationship. For a self-referential relationship that uses a second table as an association table, see the section Self-Referential Many-to-Many Relationship.

In this example, we’ll work with a single mapped class called Node, representing a tree structure:

class Node(Base):
    __tablename__ = "node"
    id = mapped_column(Integer, primary_key=True)
    parent_id = mapped_column(Integer, ForeignKey("node.id"))
    data = mapped_column(String(50))
    children = relationship("Node")
With this structure, a graph such as the following:

root --+---> child1
       +---> child2 --+--> subchild1
       |              +--> subchild2
       +---> child3
Would be represented with data such as:

id       parent_id     data
---      -------       ----
1        NULL          root
2        1             child1
3        1             child2
4        3             subchild1
5        3             subchild2
6        1             child3
The relationship() configuration here works in the same way as a “normal” one-to-many relationship, with the exception that the “direction”, i.e. whether the relationship is one-to-many or many-to-one, is assumed by default to be one-to-many. To establish the relationship as many-to-one, an extra directive is added known as relationship.remote_side, which is a Column or collection of Column objects that indicate those which should be considered to be “remote”:

class Node(Base):
    __tablename__ = "node"
    id = mapped_column(Integer, primary_key=True)
    parent_id = mapped_column(Integer, ForeignKey("node.id"))
    data = mapped_column(String(50))
    parent = relationship("Node", remote_side=[id])
Where above, the id column is applied as the relationship.remote_side of the parent relationship(), thus establishing parent_id as the “local” side, and the relationship then behaves as a many-to-one.

As always, both directions can be combined into a bidirectional relationship using two relationship() constructs linked by relationship.back_populates:

class Node(Base):
    __tablename__ = "node"
    id = mapped_column(Integer, primary_key=True)
    parent_id = mapped_column(Integer, ForeignKey("node.id"))
    data = mapped_column(String(50))
    children = relationship("Node", back_populates="parent")
    parent = relationship("Node", back_populates="children", remote_side=[id])
See also

Adjacency List - working example, updated for SQLAlchemy 2.0

Composite Adjacency Lists
A sub-category of the adjacency list relationship is the rare case where a particular column is present on both the “local” and “remote” side of the join condition. An example is the Folder class below; using a composite primary key, the account_id column refers to itself, to indicate sub folders which are within the same account as that of the parent; while folder_id refers to a specific folder within that account:

class Folder(Base):
    __tablename__ = "folder"
    __table_args__ = (
        ForeignKeyConstraint(
            ["account_id", "parent_id"], ["folder.account_id", "folder.folder_id"]
        ),
    )

    account_id = mapped_column(Integer, primary_key=True)
    folder_id = mapped_column(Integer, primary_key=True)
    parent_id = mapped_column(Integer)
    name = mapped_column(String)

    parent_folder = relationship(
        "Folder", back_populates="child_folders", remote_side=[account_id, folder_id]
    )

    child_folders = relationship("Folder", back_populates="parent_folder")
Above, we pass account_id into the relationship.remote_side list. relationship() recognizes that the account_id column here is on both sides, and aligns the “remote” column along with the folder_id column, which it recognizes as uniquely present on the “remote” side.

Self-Referential Query Strategies
Querying of self-referential structures works like any other query:

# get all nodes named 'child2'
session.scalars(select(Node).where(Node.data == "child2"))
However extra care is needed when attempting to join along the foreign key from one level of the tree to the next. In SQL, a join from a table to itself requires that at least one side of the expression be “aliased” so that it can be unambiguously referred to.

Recall from Selecting ORM Aliases in the ORM tutorial that the aliased() construct is normally used to provide an “alias” of an ORM entity. Joining from Node to itself using this technique looks like:

from sqlalchemy.orm import aliased

nodealias = aliased(Node)
session.scalars(
    select(Node)
    .where(Node.data == "subchild1")
    .join(Node.parent.of_type(nodealias))
    .where(nodealias.data == "child2")
).all()
SELECT node.id AS node_id,
        node.parent_id AS node_parent_id,
        node.data AS node_data
FROM node JOIN node AS node_1
    ON node.parent_id = node_1.id
WHERE node.data = ?
    AND node_1.data = ?
['subchild1', 'child2']
Configuring Self-Referential Eager Loading
Eager loading of relationships occurs using joins or outerjoins from parent to child table during a normal query operation, such that the parent and its immediate child collection or reference can be populated from a single SQL statement, or a second statement for all immediate child collections. SQLAlchemy’s joined and subquery eager loading use aliased tables in all cases when joining to related items, so are compatible with self-referential joining. However, to use eager loading with a self-referential relationship, SQLAlchemy needs to be told how many levels deep it should join and/or query; otherwise the eager load will not take place at all. This depth setting is configured via relationships.join_depth:

class Node(Base):
    __tablename__ = "node"
    id = mapped_column(Integer, primary_key=True)
    parent_id = mapped_column(Integer, ForeignKey("node.id"))
    data = mapped_column(String(50))
    children = relationship("Node", lazy="joined", join_depth=2)


session.scalars(select(Node)).all()
SELECT node_1.id AS node_1_id,
        node_1.parent_id AS node_1_parent_id,
        node_1.data AS node_1_data,
        node_2.id AS node_2_id,
        node_2.parent_id AS node_2_parent_id,
        node_2.data AS node_2_data,
        node.id AS node_id,
        node.parent_id AS node_parent_id,
        node.data AS node_data
FROM node
    LEFT OUTER JOIN node AS node_2
        ON node.id = node_2.parent_id
    LEFT OUTER JOIN node AS node_1
        ON node_2.id = node_1.parent_id
[]
Previous: Basic Relationship Patterns Next: Configuring how Relationship Joins
© Copyright 2007-2025, the SQLAlchemy authors and contributors.
flambé! the dragon and The Alchemist image designs created and generously donated by Rotem Yaari.