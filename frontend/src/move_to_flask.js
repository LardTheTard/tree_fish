export default async function sendMove(move_value) {
    try {
        console.log(move_value)
        const res = await fetch('/send_move', {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json', 
        },
        body: JSON.stringify(move_value) 
        });
        if (res.ok) {
            console.log('okay dokay!')
        }
    } catch (err) {
        console.error(err)
    }
}

// useEffect(() => {
//     async function load() {
//       try {
//         const res = await fetch('/api');
//         if (!res.ok) throw new Error(res.status);
//         const data = await res.text()
//         setPlaceholder(data)
//       } catch (err) {
//         setPlaceholder(err)
//       }
//     }
//     load()
//   }, []);